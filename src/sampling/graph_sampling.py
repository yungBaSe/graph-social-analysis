# src/sampling/graph_sampling.py
import pickle
import random
import networkx as nx
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Union
from tqdm import tqdm

from src.data.data_loader import get_dataset


# === PATH CONFIGURATION ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def _get_cache_path(name: str, method: str, ratio: float = None, target_size: int = None, seed: int = 42, renumber: bool = False) -> Path:
    """Stable cache key for sampled graph."""
    if target_size is not None:
        size_str = f"ts{target_size}"
    else:
        size_str = f"r{ratio:.6f}".replace(".", "_")
    renumber_str = "_renumber" if renumber else ""
    filename = f"{name}_{method}_{size_str}{renumber_str}_s{seed}.pkl"
    return DATA_PROCESSED / filename


def random_node_sampling(G: nx.Graph | nx.DiGraph, ratio: float, seed: int = 42) -> nx.Graph | nx.DiGraph:
    """Random node sampling (baseline)."""
    random.seed(seed)
    n = G.number_of_nodes()
    k = int(n * ratio)
    nodes = random.sample(list(G.nodes()), k)
    return G.subgraph(nodes).copy()


def random_edge_sampling(G: nx.Graph | nx.DiGraph, ratio: float, seed: int = 42) -> nx.Graph | nx.DiGraph:
    """Random Edge sampling."""
    random.seed(seed)
    edges = list(G.edges())
    k = int(len(edges) * ratio)
    sampled_edges = random.sample(edges, min(k, len(edges)))
    H = G.__class__()
    H.add_edges_from(sampled_edges)
    return H


def forest_fire_sampling(G: nx.Graph | nx.DiGraph, ratio: float, p: float = 0.7, seed: int = 42) -> nx.Graph | nx.DiGraph:
    """Forest Fire sampling — хорошо сохраняет структуру."""
    random.seed(seed)
    n = G.number_of_nodes()
    target = int(n * ratio)
    start = random.choice(list(G.nodes()))
    sampled = {start}
    queue = [start]

    while len(sampled) < target and queue:
        u = queue.pop(0)
        for v in G.neighbors(u):
            if v not in sampled and random.random() < p:
                sampled.add(v)
                queue.append(v)
                if len(sampled) >= target:
                    break

    H = G.__class__()
    H.add_nodes_from(sampled)
    for u in sampled:
        for v in G.neighbors(u):
            if v in sampled:
                H.add_edge(u, v)
    return H


def snowball_sampling(G: nx.Graph | nx.DiGraph, ratio: float, seed: int = 42) -> nx.Graph | nx.DiGraph:
    """Snowball / BFS sampling."""
    random.seed(seed)
    n = G.number_of_nodes()
    target = int(n * ratio)
    start = random.choice(list(G.nodes()))
    sampled = {start}
    queue = [start]

    while len(sampled) < target and queue:
        u = queue.pop(0)
        for v in G.neighbors(u):
            if v not in sampled:
                sampled.add(v)
                queue.append(v)
                if len(sampled) >= target:
                    break

    H = G.__class__()
    H.add_nodes_from(sampled)
    for u in sampled:
        for v in G.neighbors(u):
            if v in sampled:
                H.add_edge(u, v)
    return H


import random
import networkx as nx

def ties_sampling(
    G: nx.Graph,
    target_num_nodes: int,
    seed: int = 42,
    max_iter: int = 25,
    tolerance: float = 0.02  # 2% допустимое отклонение
) -> nx.Graph:
    """
    Строит TIES-подграф (упрощённая версия со случайным отбором рёбер)
    с желаемым ЧИСЛОМ ВЕРШИН target_num_nodes.

    Фазы:
      1. Случайный отбор доли p от всех рёбер.
      2. Индукция: все рёбра исходного графа между вершинами, попавшими в выборку.
    Параметр p подбирается бинарным поиском.
    """
    random.seed(seed)
    edges = list(G.edges())
    n_edges = len(edges)
    target = target_num_nodes
    total_nodes = G.number_of_nodes()

    if target >= total_nodes:
        return G.copy()

    low, high = 0.0, 1.0
    best_H = None
    best_diff = float('inf')

    for _ in range(max_iter):
        mid = (low + high) / 2
        k = int(n_edges * mid)
        if k == 0:
            low = mid
            continue
        # Фаза 1: отбираем k случайных рёбер
        sampled_edges = random.sample(edges, min(k, n_edges))
        # Извлекаем множество вершин
        nodes = set()
        for u, v in sampled_edges:
            nodes.add(u)
            nodes.add(v)
        # Фаза 2: индукция — строим индуцированный подграф на этих вершинах
        H = G.subgraph(nodes).copy()
        actual_nodes = H.number_of_nodes()
        diff = abs(actual_nodes - target)

        if actual_nodes == target:
            return H
        if diff < best_diff:
            best_diff = diff
            best_H = H

        if actual_nodes < target:
            low = mid   # нужно больше рёбер -> больше p
        else:
            high = mid  # нужно меньше рёбер -> меньше p

    # Если не попали точно, возвращаем лучшее
    if best_H is None:
        # fallback
        return G.subgraph(list(G.nodes())[:target]).copy()
    return best_H


def sample_graph(
    G: nx.Graph | nx.DiGraph,
    method: str = "forest_fire",
    ratio: Optional[float] = None,
    target_size: Optional[int] = None,
    seed: int = 42,
    name: Optional[str] = None,
    renumber: bool = False,
) -> Dict[str, Any]:
    """Main sampling function. Returns rich dict with mapping and stats."""
    if ratio is None and target_size is None:
        ratio = 0.1
    if target_size is not None:
        ratio = target_size / G.number_of_nodes()

    cache_path = None
    if name:
        cache_path = _get_cache_path(name, method, ratio, target_size, seed, renumber)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

    methods = {
        "random_node": random_node_sampling,
        "random_edge": random_edge_sampling,
        "forest_fire": forest_fire_sampling,
        "snowball": snowball_sampling,
        "ties": ties_sampling,
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")

    sampled_graph = methods[method](G, ratio=ratio, seed=seed)

    # === Mapping & renumbering ===
    original_nodes = list(G.nodes())
    sampled_nodes = list(sampled_graph.nodes())

    if renumber:
        node_map = {old: new for new, old in enumerate(sorted(sampled_nodes))}
        H = nx.relabel_nodes(sampled_graph, node_map, copy=True)
        mapping = {old: node_map[old] for old in sampled_nodes}  # old → new
        inverse_mapping = {v: k for k, v in mapping.items()}
    else:
        H = sampled_graph.copy()
        mapping = {n: n for n in sampled_nodes}
        inverse_mapping = mapping.copy()

    result = {
        "graph": H,
        "mapping": mapping,
        "inverse_mapping": inverse_mapping,
        "original_n_nodes": G.number_of_nodes(),
        "sampled_n_nodes": H.number_of_nodes(),
        "original_n_edges": G.number_of_edges(),
        "sampled_n_edges": H.number_of_edges(),
        "method": method,
        "ratio": ratio,
        "seed": seed,
        "is_directed": isinstance(G, nx.DiGraph),
        "renumbered": renumber,
    }

    if name and cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"✅ Sample cached: {name} | {method} | ratio={ratio:.3f} | nodes={H.number_of_nodes():,} | renumber={renumber}")

    return result


def sample_dataset(
    name: str,
    method: str = "forest_fire",
    ratio: Optional[float] = None,
    target_size: Optional[int] = None,
    seed: int = 42,
    renumber: bool = False,
) -> Dict[str, Any]:
    """Convenience wrapper."""
    dataset = get_dataset(name, verbose=False)
    return sample_graph(
        G=dataset["graph"],
        method=method,
        ratio=ratio,
        target_size=target_size,
        seed=seed,
        name=name,
        renumber=renumber,
    )


def list_cached_samples() -> None:
    """List all cached sampled graphs."""
    files = sorted(DATA_PROCESSED.glob("*_r*_s*.pkl")) + sorted(DATA_PROCESSED.glob("*_ts*_s*.pkl"))
    print(f"Found {len(files)} cached samples:")
    for f in files:
        print(f"  • {f.name}")