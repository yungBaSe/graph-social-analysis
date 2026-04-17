# src/embeddings/graph_embeddings.py (обновлённая версия с seed)
"""
Модуль для построения эмбеддингов узлов графа.
Использует gensim для Node2Vec/DeepWalk и чистый PyTorch для GraphSAGE.
"""

import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import networkx as nx
from tqdm import tqdm

# ----------------------------- Конфигурация -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- Node2Vec / DeepWalk (через gensim) -----------------------------
try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("⚠️ gensim не установлен. Установите: pip install gensim")


def generate_random_walks(
    G: nx.Graph,
    num_walks: int = 10,
    walk_length: int = 80,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = 42
) -> List[List[str]]:
    """
    Генерирует случайные блуждания по графу (Node2Vec стиль).
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Приводим к неориентированному
    if G.is_directed():
        G = G.to_undirected()
    
    nodes = list(G.nodes())
    walks = []
    
    # Предварительно считаем переходные вероятности для Node2Vec
    if p != 1.0 or q != 1.0:
        # Строим алиасы для ускорения
        alias_nodes = {}
        for node in nodes:
            neighbors = list(G.neighbors(node))
            if len(neighbors) > 0:
                probs = [1.0 / len(neighbors)] * len(neighbors)
                alias_nodes[node] = _alias_setup(probs)
            else:
                alias_nodes[node] = None
        
        alias_edges = {}
        for u, v in G.edges():
            alias_edges[(u, v)] = _get_alias_edge(G, u, v, p, q)
            alias_edges[(v, u)] = _get_alias_edge(G, v, u, p, q)
    else:
        alias_nodes = None
        alias_edges = None
    
    for _ in tqdm(range(num_walks), desc="Генерация блужданий"):
        random.shuffle(nodes)
        for start_node in nodes:
            if p == 1.0 and q == 1.0:
                # DeepWalk — просто случайные блуждания
                walk = [str(start_node)]
                current = start_node
                for _ in range(walk_length - 1):
                    neighbors = list(G.neighbors(current))
                    if not neighbors:
                        break
                    current = random.choice(neighbors)
                    walk.append(str(current))
                walks.append(walk)
            else:
                # Node2Vec с p и q
                walk = _node2vec_walk(G, start_node, walk_length, 
                                      alias_nodes, alias_edges)
                walks.append([str(n) for n in walk])
    
    return walks


def _alias_setup(probs: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Настройка алиас-метода для быстрой дискретной выборки."""
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=int)
    
    smaller = []
    larger = []
    for i, prob in enumerate(probs):
        q[i] = K * prob
        if q[i] < 1.0:
            smaller.append(i)
        else:
            larger.append(i)
    
    while smaller and larger:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    
    return J, q


def _alias_draw(J: np.ndarray, q: np.ndarray, K: int) -> int:
    """Выборка из алиас-таблицы."""
    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def _get_alias_edge(G: nx.Graph, u: int, v: int, p: float, q: float):
    """Построение алиас-таблицы для рёбер в Node2Vec."""
    neighbors = list(G.neighbors(v))
    probs = []
    for x in neighbors:
        if x == u:
            probs.append(1.0 / p)
        elif G.has_edge(x, u):
            probs.append(1.0)
        else:
            probs.append(1.0 / q)
    
    # Нормализация
    total = sum(probs)
    probs = [prob / total for prob in probs]
    
    return _alias_setup(probs)


def _node2vec_walk(G: nx.Graph, start_node: int, walk_length: int,
                   alias_nodes: Dict, alias_edges: Dict) -> List[int]:
    """Одно блуждание Node2Vec."""
    walk = [start_node]
    
    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(G.neighbors(cur))
        if not cur_nbrs:
            break
        
        if len(walk) == 1:
            # Первый шаг — из стартового узла
            J, q = alias_nodes[cur]
            idx = _alias_draw(J, q, len(cur_nbrs))
            walk.append(cur_nbrs[idx])
        else:
            # Последующие шаги — учитываем предыдущий узел
            prev = walk[-2]
            J, q = alias_edges[(prev, cur)]
            idx = _alias_draw(J, q, len(cur_nbrs))
            walk.append(cur_nbrs[idx])
    
    return walk


def train_node2vec_gensim(
    G: nx.Graph,
    graph_name: str = "graph",
    dimensions: int = 128,
    walk_length: int = 80,
    num_walks: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    window: int = 10,
    min_count: int = 1,
    workers: int = 4,
    epochs: int = 5,
    save_model: bool = True,
    model_name: Optional[str] = None,
    seed: int = 42  # <-- добавлен параметр seed
) -> Tuple[Dict[int, np.ndarray], np.ndarray, any]:
    """
    Обучает Node2Vec через gensim.models.Word2Vec.
    """
    if not GENSIM_AVAILABLE:
        raise ImportError("gensim не установлен")
    
    if model_name is None:
        model_name = f"{graph_name}_node2vec_d{dimensions}_p{p}_q{q}"
    
    emb_path = EMBEDDINGS_DIR / f"{model_name}_embeddings.pkl"
    model_path = MODELS_DIR / f"{model_name}.model"
    
    if emb_path.exists():
        print(f"📂 Загрузка готовых эмбеддингов из {emb_path}")
        with open(emb_path, 'rb') as f:
            data = pickle.load(f)
        return data['dict'], data['matrix'], None
    
    print(f"🚀 Генерация блужданий для Node2Vec...")
    walks = generate_random_walks(G, num_walks, walk_length, p, q, seed=seed)
    
    print(f"🚀 Обучение Word2Vec: dim={dimensions}, window={window}")
    model = Word2Vec(
        sentences=walks,
        vector_size=dimensions,
        window=window,
        min_count=min_count,
        sg=1,  # Skip-gram
        workers=workers,
        epochs=epochs,
        seed=seed  # <-- seed для Word2Vec
    )
    
    # Извлекаем эмбеддинги
    np.random.seed(seed)
    embeddings_dict = {}
    for node in G.nodes():
        node_str = str(node)
        if node_str in model.wv:
            embeddings_dict[node] = model.wv[node_str]
        else:
            # Для изолированных узлов — случайный вектор
            embeddings_dict[node] = np.random.randn(dimensions) * 0.01
    
    nodes_order = list(G.nodes())
    embeddings_matrix = np.array([embeddings_dict[n] for n in nodes_order])
    
    if save_model:
        model.save(str(model_path))
        with open(emb_path, 'wb') as f:
            pickle.dump({'dict': embeddings_dict, 'matrix': embeddings_matrix}, f)
        print(f"💾 Модель и эмбеддинги сохранены: {model_name}")
    
    return embeddings_dict, embeddings_matrix, model


def train_deepwalk_gensim(G: nx.Graph, graph_name: str = "graph", **kwargs) -> Tuple:
    """DeepWalk = Node2Vec с p=1, q=1."""
    kwargs['p'] = 1.0
    kwargs['q'] = 1.0
    return train_node2vec_gensim(G, graph_name, **kwargs)


# ----------------------------- GraphSAGE (чистый PyTorch) -----------------------------
# Отложенный импорт PyTorch — только при реальном вызове функции
TORCH_AVAILABLE = False
_IMPORT_ERROR_MSG = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERROR_MSG = str(e)
    print(f"⚠️ PyTorch не установлен или повреждён. GraphSAGE будет недоступен. Ошибка: {e}")
except AttributeError as e:
    _IMPORT_ERROR_MSG = str(e)
    print(f"⚠️ PyTorch повреждён (циклический импорт). Переустановите: pip uninstall torch -y && pip install torch")
except Exception as e:
    _IMPORT_ERROR_MSG = str(e)
    print(f"⚠️ Неизвестная ошибка при импорте PyTorch: {e}")


class GraphSAGEModel:
    """Заглушка для GraphSAGE (реальная модель создаётся только при TORCH_AVAILABLE=True)."""
    pass

def train_graphsage_torch(
    G: nx.Graph,
    graph_name: str = "graph",
    dimensions: int = 128,
    hidden_dims: List[int] = [256, 256],
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu',
    save_model: bool = True,
    model_name: Optional[str] = None,
    seed: int = 42
) -> Tuple[Dict[int, np.ndarray], np.ndarray, any]:
    """
    Обучает GraphSAGE на задаче восстановления связей (link prediction).
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            f"PyTorch не доступен. GraphSAGE нельзя использовать.\n"
            f"Причина: {_IMPORT_ERROR_MSG}\n"
            f"Установите PyTorch: pip install torch"
        )
    
    # Реальная реализация GraphSAGE (выполнится только если TORCH_AVAILABLE=True)
    # Устанавливаем сиды для воспроизводимости
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if model_name is None:
        model_name = f"{graph_name}_graphsage_d{dimensions}_h{'-'.join(map(str,hidden_dims))}"
    
    emb_path = EMBEDDINGS_DIR / f"{model_name}_embeddings.pkl"
    if emb_path.exists():
        print(f"📂 Загрузка готовых эмбеддингов из {emb_path}")
        with open(emb_path, 'rb') as f:
            data = pickle.load(f)
        return data['dict'], data['matrix'], None
    
    # Приводим к неориентированному
    if G.is_directed():
        G = G.to_undirected()
    
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    # Матрица смежности
    adj = nx.to_numpy_array(G)
    adj = torch.FloatTensor(adj).to(device)
    
    # Признаки узлов — one-hot или degree
    degrees = np.array([G.degree(n) for n in nodes])
    max_deg = degrees.max()
    features = np.zeros((n_nodes, max_deg + 1))
    features[np.arange(n_nodes), degrees] = 1.0
    features = torch.FloatTensor(features).to(device)
    
    # Определяем класс модели внутри функции, чтобы избежать проблем с отсутствием torch.nn
    class _GraphSAGEModel(nn.Module):
        def __init__(self, in_feats: int, hidden_feats: List[int], out_feats: int):
            super().__init__()
            self.layers = nn.ModuleList()
            
            self.layers.append(self._make_layer(in_feats, hidden_feats[0]))
            
            # Скрытые слои
            for i in range(len(hidden_feats) - 1):
                self.layers.append(self._make_layer(hidden_feats[i], hidden_feats[i+1]))
            
            self.out_layer = nn.Linear(hidden_feats[-1], out_feats)
        
        def _make_layer(self, in_dim: int, out_dim: int):
            """Слой с агрегацией соседей (конкатенация -> удвоение размерности)."""
            return nn.Sequential(
                nn.Linear(in_dim * 2, out_dim),
                nn.ReLU(),
                nn.BatchNorm1d(out_dim)
            )
        
        def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                neighbor_agg = torch.mm(adj, x) / (adj.sum(dim=1, keepdim=True) + 1e-10)
                combined = torch.cat([x, neighbor_agg], dim=1)
                x = layer(combined)
            
            # Выходной слой без агрегации соседей
            return self.out_layer(x)
    
    model = _GraphSAGEModel(
        in_feats=features.shape[1],
        hidden_feats=hidden_dims,
        out_feats=dimensions
    ).to(device)
    model.max_deg = max_deg
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    edges = list(G.edges())
    non_edges = list(nx.non_edges(G))
    
    print(f"🚀 Обучение GraphSAGE: {epochs} эпох")
    model.train()
    
    for epoch in range(epochs):
        pos_sample = random.sample(edges, min(len(edges), 1024))
        neg_sample = random.sample(non_edges, len(pos_sample))
        
        optimizer.zero_grad()
        emb = model(features, adj)
        
        pos_scores = (emb[[node_to_idx[u] for u, _ in pos_sample]] * 
                      emb[[node_to_idx[v] for _, v in pos_sample]]).sum(dim=1)
        neg_scores = (emb[[node_to_idx[u] for u, _ in neg_sample]] * 
                      emb[[node_to_idx[v] for _, v in neg_sample]]).sum(dim=1)
        
        loss = -torch.log(torch.sigmoid(pos_scores) + 1e-10).mean() - \
                torch.log(1 - torch.sigmoid(neg_scores) + 1e-10).mean()
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        embeddings_matrix = model(features, adj).cpu().numpy()
    
    embeddings_dict = {nodes[i]: embeddings_matrix[i] for i in range(n_nodes)}
    
    if save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'max_deg': max_deg
        }, MODELS_DIR / f"{model_name}.pt")
        with open(emb_path, 'wb') as f:
            pickle.dump({'dict': embeddings_dict, 'matrix': embeddings_matrix}, f)
        print(f"💾 Модель и эмбеддинги сохранены: {model_name}")
    
    return embeddings_dict, embeddings_matrix, model


# ----------------------------- Универсальная обёртка ----------------------------
def get_embeddings(
    G: nx.Graph,
    graph_name: str = "graph",
    method: str = 'node2vec',
    **params
) -> Tuple[Dict[int, np.ndarray], np.ndarray, any]:
    """
    Универсальная функция для получения эмбеддингов.
    
    Параметры
    ---------
    method : {'node2vec', 'deepwalk', 'graphsage'}
    """
    if method.lower() == 'node2vec':
        return train_node2vec_gensim(G, graph_name=graph_name, **params)
    elif method.lower() == 'deepwalk':
        return train_deepwalk_gensim(G, graph_name=graph_name, **params)
    elif method.lower() == 'graphsage':
        return train_graphsage_torch(G, graph_name=graph_name, **params)
    else:
        raise ValueError(f"Неизвестный метод: {method}")