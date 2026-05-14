# src/data/data_loader.py
import pickle
import gzip
import zipfile
import json
import tarfile
import requests
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
from collections import defaultdict
import scipy.sparse as sp

import torch
import torch_geometric.data.data
torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage,
])

# === PATH CONFIGURATION ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTRACTED = PROJECT_ROOT / "data" / "extracted"
PYG_DATA = PROJECT_ROOT / "data" / "pyg"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
DATA_EXTRACTED.mkdir(parents=True, exist_ok=True)
PYG_DATA.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = Path(__file__).parent / "datasets.json"


def load_datasets_config() -> dict:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    for name, info in config.items():
        if info.get("type") == "zip" and "extract_subdir" in info:
            info["extract_path"] = DATA_EXTRACTED / info["extract_subdir"]
        if info.get("root_subdir"):
            info["root"] = PYG_DATA / info["root_subdir"]
    return config


DATASETS = load_datasets_config()


def download_file(url: str, local_path: Path) -> None:
    if local_path.exists():
        return
    print(f"Downloading {local_path.name}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.namelist():
            filename = Path(member).name
            if not filename:
                continue
            target_path = extract_to / filename
            if not target_path.exists():
                z.extract(member, extract_to)
                temp_path = extract_to / member
                if temp_path.exists() and temp_path != target_path:
                    temp_path.rename(target_path)


# ------------------------------------------------------------
#  Универсальные загрузчики для разных форматов
# ------------------------------------------------------------
def load_graph_from_gz(info: dict) -> nx.Graph:
    download_file(info["url"], DATA_RAW / info["filename"])
    gz_path = DATA_RAW / info["filename"]
    G = nx.DiGraph() if info.get("is_directed", False) else nx.Graph()
    with gzip.open(gz_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                u, v = map(int, parts[:2])
                G.add_edge(u, v)
    return G


def load_graph_from_zip(info: dict) -> dict:
    download_file(info["url"], DATA_RAW / info["filename"])
    zip_path = DATA_RAW / info["filename"]
    extract_zip(zip_path, info["extract_path"])

    edges_path = info["extract_path"] / info["edges_file"]
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_path}")

    G = nx.DiGraph() if info.get("is_directed", False) else nx.Graph()
    with open(edges_path, 'r') as f:
        header = f.readline().strip()
        if 'from' in header.lower() or 'node' in header.lower():
            pass
        else:
            f.seek(0)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                G.add_edge(u, v)

    features = None
    labels = None

    if "features_file" in info:
        feat_path = info["extract_path"] / info["features_file"]
        if feat_path.exists():
            fmt = info.get("features_format", "json")
            if fmt == "onehot_json":
                # Передаём отсортированный список узлов графа
                nodes = sorted(G.nodes())
                features = load_onehot_json_features(feat_path, nodes=nodes)
            elif fmt == "csv":
                features = load_features_from_csv(feat_path)
            else:
                features = load_features_from_json(feat_path)

    if "labels_file" in info:
        label_path = info["extract_path"] / info["labels_file"]
        if label_path.exists():
            index_col = info.get("labels_index_col", 0)
            labels = load_labels_from_csv(label_path, index_col=index_col)
            # Если указан target_column, оставляем только его (как DataFrame)
            target_col = info.get("target_column")
            if target_col and target_col in labels.columns:
                labels = labels[[target_col]]

    return {"graph": G, "features": features, "labels": labels}


def load_pyg_planetoid(info: dict) -> dict:
    from torch_geometric.datasets import Planetoid
    name = info["pyg_name"]
    dataset = Planetoid(root=str(PYG_DATA), name=name)
    data = dataset[0]
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    edge_index = data.edge_index.numpy()
    G.add_edges_from(edge_index.T)
    features = pd.DataFrame(data.x.numpy(), index=np.arange(data.num_nodes))
    labels = pd.DataFrame(data.y.numpy(), columns=['label'])
    labels.index = np.arange(data.num_nodes)
    return {"graph": G, "features": features, "labels": labels}


def load_pyg_dataset(info: dict) -> dict:
    import importlib
    module = importlib.import_module(info["pyg_module"])
    DatasetClass = getattr(module, info["pyg_class"])
    dataset = DatasetClass(root=str(PYG_DATA))
    data = dataset[0]
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    edge_index = data.edge_index.numpy()
    G.add_edges_from(edge_index.T)
    features = pd.DataFrame(data.x.numpy(), index=np.arange(data.num_nodes))
    labels = pd.DataFrame(data.y.numpy(), columns=['label'])
    labels.index = np.arange(data.num_nodes)
    return {"graph": G, "features": features, "labels": labels}


def load_ogb_dataset(info: dict) -> dict:
    from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
    root_path = info["root"]
    dataset = PygNodePropPredDataset(name=info["pyg_dataset_name"], root=str(root_path))
    data = dataset[0]
    G = nx.Graph()
    n_nodes = data.num_nodes
    G.add_nodes_from(range(n_nodes))
    edge_index = data.edge_index.numpy()
    G.add_edges_from(edge_index.T)
    features = pd.DataFrame(data.x.numpy(), index=np.arange(n_nodes))
    labels_np = data.y.numpy().flatten()
    labels = pd.DataFrame(labels_np, columns=['label'])
    labels.index = np.arange(n_nodes)
    masks = {}
    if hasattr(data, 'train_mask'):
        masks['train_mask'] = data.train_mask.numpy()
        masks['val_mask'] = data.val_mask.numpy()
        masks['test_mask'] = data.test_mask.numpy()
    return {"graph": G, "features": features, "labels": labels, "masks": masks}


def load_onehot_json_features(path: Path, nodes: Optional[list] = None) -> sp.csr_matrix:
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Если список узлов не задан, берём все ключи из JSON
    if nodes is None:
        nodes = sorted(map(int, data.keys()))
    n_nodes = len(nodes)
    
    # Определяем размерность признаков (максимальный индекс + 1)
    max_idx = 0
    for node in nodes:
        indices = data.get(str(node), [])
        if indices:
            max_idx = max(max_idx, max(indices))
    n_features = max_idx + 1
    
    rows, cols = [], []
    for i, node in enumerate(nodes):
        indices = data.get(str(node), [])
        for idx in indices:
            rows.append(i)
            cols.append(idx)
    
    values = np.ones(len(rows), dtype=np.float32)
    features = sp.csr_matrix((values, (rows, cols)), shape=(n_nodes, n_features))
    return features


def load_features_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(int)
    df = df.sort_index()
    return df


def load_features_from_json(path: Path) -> pd.DataFrame:
    with open(path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index = df.index.astype(int)
    df = df.sort_index()
    return df


def load_labels_from_csv(path: Path, index_col=0) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=index_col)
    df.index = df.index.astype(int)
    df = df.sort_index()
    return df


# ------------------------------------------------------------
#  Ego-сети Facebook / Twitter
# ------------------------------------------------------------
def parse_ego_edges(filepath) -> nx.Graph:
    G = nx.Graph()
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                G.add_edge(u, v)
    return G


def parse_ego_circles(filepath) -> dict:
    circles = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                circle_name = parts[0]
                members = list(map(int, parts[1:]))
                circles[circle_name] = members
    return circles


def parse_ego_features(feat_filepath, featnames_filepath, egofeat_filepath=None):
    # Читаем названия признаков
    names = []
    with open(featnames_filepath, 'r') as f:
        for line in f:
            name = line.strip().split(' ', 1)[1] if ' ' in line else line.strip()
            names.append(name)

    # Фичи для друзей
    node_feats = {}
    with open(feat_filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            node_id = int(parts[0])
            values = list(map(int, parts[1:]))
            node_feats[node_id] = dict(zip(names, values))

    # Фичи эго-узла
    ego_feats = None
    if egofeat_filepath and Path(egofeat_filepath).exists():
        with open(egofeat_filepath, 'r') as f:
            line = f.readline().strip()
            if line:
                vals = list(map(int, line.split()))
                ego_feats = dict(zip(names, vals))

    return node_feats, ego_feats


def load_ego_network(info: dict) -> dict:
    """Загружает ego-сеть: граф может быть из combined-файла, фичи/круги из архива."""
    # --- Загрузка графа ---
    if "combined_url" in info:
        # Используем комбинированный граф (например, Facebook)
        download_file(info["combined_url"], DATA_RAW / info["combined_filename"])
        combined_path = DATA_RAW / info["combined_filename"]
        G = nx.Graph()  # Facebook неориентированный
        with gzip.open(combined_path, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    G.add_edge(u, v)
    else:
        # Только из архива (Twitter)
        G = nx.DiGraph() if info.get("is_directed", False) else nx.Graph()

    # --- Фичи и круги из архива ---
    download_file(info["url"], DATA_RAW / info["filename"])
    archive_path = DATA_RAW / info["filename"]
    extract_dir = DATA_EXTRACTED / info["extract_subdir"]

    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_dir)

    edge_files = list(extract_dir.rglob("*.edges"))
    all_circles = defaultdict(dict)
    feature_dict = {}
    feat_names = None

    for f in edge_files:
        ego = int(f.stem)
        base_dir = f.parent

        # Если граф не был загружен из combined, добавляем рёбра из этого ego-файла
        if "combined_url" not in info:
            ego_G = parse_ego_edges(f)
            G.add_edges_from(ego_G.edges())

        # Круги
        cf = base_dir / f"{ego}.circles"
        if cf.exists():
            circles = parse_ego_circles(cf)
            if circles:
                all_circles[ego] = circles

        # Фичи
        featf = base_dir / f"{ego}.feat"
        namesf = base_dir / f"{ego}.featnames"
        egof = base_dir / f"{ego}.egofeat"
        if featf.exists() and namesf.exists():
            node_feats, ego_feats = parse_ego_features(featf, namesf, egof)
            if feat_names is None and node_feats:
                feat_names = list(next(iter(node_feats.values())).keys())
            for nid, fv in node_feats.items():
                if nid not in feature_dict:
                    feature_dict[nid] = fv
            if ego_feats:
                feature_dict[ego] = ego_feats

    features_df = None
    if feature_dict:
        df = pd.DataFrame.from_dict(feature_dict, orient='index', columns=feat_names, dtype=np.float32)
        features_df = df.sort_index()

    return {
        "graph": G,
        "features": features_df,
        "circles": dict(all_circles) if all_circles else None
    }

def load_youtube_ground_truth(info: dict) -> dict:
    """Загружает полный граф YouTube и список всех сообществ (top5000)."""
    # Граф
    download_file(info["url"], DATA_RAW / info["filename"])
    gz_path = DATA_RAW / info["filename"]
    G = nx.Graph()
    with gzip.open(gz_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            u, v = map(int, line.strip().split())
            G.add_edge(u, v)

    # Сообщества
    download_file(info["cmty_url"], DATA_RAW / info["cmty_filename"])
    cmty_path = DATA_RAW / info["cmty_filename"]
    communities = []
    with gzip.open(cmty_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            nodes = list(map(int, line.strip().split()))
            if len(nodes) >= 3:
                communities.append(nodes)

    # Сортируем по размеру (от крупных к мелким) для удобства
    communities.sort(key=len, reverse=True)

    return {"graph": G, "all_communities": communities}


# ------------------------------------------------------------
#  Основная функция get_dataset
# ------------------------------------------------------------
def get_dataset(name: str, verbose: bool = False) -> dict:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")

    full_pkl = DATA_PROCESSED / f"{name}_full.pkl"
    if full_pkl.exists():
        with open(full_pkl, 'rb') as f:
            dataset = pickle.load(f)
        if verbose:
            describe_dataset(dataset)
        return dataset

    info = DATASETS[name]
    graph = None
    features = None
    labels = None
    circles = None
    masks = None

    if info["type"] == "gz":
        graph = load_graph_from_gz(info)
    elif info["type"] == "zip":
        tmp = load_graph_from_zip(info)
        graph = tmp["graph"]
        features = tmp.get("features")
        labels = tmp.get("labels")
    elif info["type"] == "pyg_planetoid":
        tmp = load_pyg_planetoid(info)
        graph = tmp["graph"]
        features = tmp["features"]
        labels = tmp["labels"]
    elif info["type"] == "pyg_dataset":
        tmp = load_pyg_dataset(info)
        graph = tmp["graph"]
        features = tmp["features"]
        labels = tmp["labels"]
    elif info["type"] == "pyg_ogb":
        tmp = load_ogb_dataset(info)
        graph = tmp["graph"]
        features = tmp["features"]
        labels = tmp["labels"]
        masks = tmp.get("masks")
    elif info["type"] == "ego_network":
        tmp = load_ego_network(info)
        graph = tmp["graph"]
        features = tmp["features"]
        circles = tmp["circles"]
    elif info["type"] == "youtube_ground_truth":
        tmp = load_youtube_ground_truth(info)
        graph = tmp["graph"]
        dataset = {
            "graph": graph,
            "features": None,
            "labels": None,
            "circles": None,
            "masks": None,
            "name": name,
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "is_directed": False,
            "all_communities": tmp["all_communities"],  # сохраняем!
        }
        # кэшируем сразу
        with open(full_pkl, 'wb') as f:
            pickle.dump(dataset, f)
        if verbose:
            describe_dataset(dataset)
        return dataset
    else:
        raise ValueError(f"Unknown dataset type: {info['type']}")

    # Догрузка фич для датасетов с external features (например, Pokec)
    if features is None and info.get("features_available"):
        # Такого типа сейчас только pokec, который всё ещё zip без features_file
        if name == "pokec":
            features = load_features(name, sorted(graph.nodes()))

    dataset = {
        "graph": graph,
        "features": features,
        "labels": labels,
        "circles": circles,
        "masks": masks,
        "name": name,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "is_directed": info.get("is_directed", False),
        "target_column": info.get("target_column"),
    }

    with open(full_pkl, 'wb') as f:
        pickle.dump(dataset, f)
    if verbose:
        describe_dataset(dataset)
    return dataset


def describe_dataset(dataset: dict) -> None:
    G = dataset["graph"]
    print(f"\n{'='*70}")
    print(f"DATASET SUMMARY: {dataset['name'].upper()}")
    print(f"{'='*70}")
    print(f"Type           : {'Directed' if dataset['is_directed'] else 'Undirected'}")
    print(f"Nodes          : {dataset['n_nodes']:,}")
    print(f"Edges          : {dataset['n_edges']:,}")
    if dataset.get("features") is not None:
        f = dataset["features"]
        if hasattr(f, 'shape'):
            print(f"Node features  : {f.shape[1]:,} dims")
        else:
            print(f"Node features  : available")
    if dataset.get("labels") is not None:
        labels = dataset["labels"].iloc[:, 0]
        print(f"Node labels    : {labels.nunique()} classes")
    if dataset.get("circles") is not None:
        total = sum(len(circle_dict) for circle_dict in dataset["circles"].values())
        print(f"Circles        : {total} circle sets across {len(dataset['circles'])} ego nodes")
    if dataset.get("masks") is not None:
        print("Data splits    : OGB predefined masks")
    print(f"{'='*70}\n")


def list_available_datasets() -> None:
    print("Available datasets:")
    for k, v in DATASETS.items():
        print(f"  • {k:12} — {v['description']} ({'directed' if v['is_directed'] else 'undirected'})")


def load_features(name: str, nodes: list) -> Optional[pd.DataFrame]:
    info = DATASETS.get(name)
    if not info or not info.get("features_available"):
        return None
    if name == "pokec":
        features_path = DATA_RAW / "soc-pokec-profiles.txt.gz"
        if not features_path.exists():
            download_file("https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz", features_path)
        print("Loading Pokec profiles...")
        df = pd.read_csv(
            features_path, sep='\t', header=None,
            names=[
                "user_id", "public", "completion_percentage", "gender", "region",
                "last_login", "registration", "AGE", "body", "I_am_working_in_field",
                "spoken_languages", "hobbies", "I_most_enjoy_good_food", "pets",
                "body_type", "my_eyesight", "eye_color", "hair_color", "hair_type",
                "completed_level_of_education", "favourite_color", "relation_to_smoking",
                "relation_to_alcohol", "sign_in_zodiac", "on_pokec_i_am_looking_for",
                "love_is_for_me", "relation_to_casual_sex", "my_partner_should_be",
                "marital_status", "children", "relation_to_children", "I_like_movies",
                "I_like_watching_movie", "I_like_music", "I_mostly_like_listening_to_music",
                "the_idea_of_good_evening", "I_like_specialties_from_kitchen", "fun",
                "I_am_going_to_concerts", "my_active_sports", "my_passive_sports",
                "profession", "I_like_books", "life_style", "music", "cars", "politics",
                "relationships", "art_culture", "hobbies_interests",
                "science_technologies", "computers_internet", "education", "sport",
                "movies", "travelling", "health", "companies_brands", "more"
            ],
            dtype=str, na_values=["null"], low_memory=False, encoding='utf-8'
        )
        numeric_cols = ["public", "completion_percentage", "gender", "AGE", "body", "height", "weight",
                        "I_am_working_in_field", "eye_color", "hair_color", "smoking_habits", "drinking_habits",
                        "how_often_i_exercise", "my_education", "monthly_income"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Устанавливаем индекс
        df = df.set_index("user_id")
        
        # Явно приводим индексы к np.int64 для совместимости с узлами графа
        df.index = df.index.astype(np.int64)
        
        # Удаляем дубликаты индекса (оставляем последнюю запись)
        before = len(df)
        df = df[~df.index.duplicated(keep='last')]
        after = len(df)
        if before != after:
            print(f"⚠️ Removed {before - after} duplicate user_id entries (kept last occurrence)")
        
        # reindex с явным numpy int64 массивом узлов
        return df.reindex(np.array(nodes, dtype=np.int64))
    return None


def reload_datasets() -> None:
    global DATASETS
    DATASETS = load_datasets_config()
    print("Datasets config reloaded.")