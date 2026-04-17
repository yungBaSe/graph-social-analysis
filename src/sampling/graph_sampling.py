# src/sampling/graph_sampling.py
"""
Модуль для сэмплирования (создания подграфов) больших графов.

Поддерживаемые методы:
- random_node: случайный выбор узлов и индуцированный подграф
- random_edge: случайный выбор рёбер и индуцированный подграф
- snowball: BFS-семплирование (снежный ком)
- forest_fire: Forest Fire семплирование
- random_walk: семплирование на основе случайных блужданий
- mhrw: Metropolis-Hastings Random Walk

Все функции возвращают networkx.Graph и поддерживают кеширование.
"""

import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

# ----------------------------- Конфигурация -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = PROJECT_ROOT / "data" / "processed" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def _get_sample_path(method: str, params: Dict) -> Path:
    """Генерирует путь для сохранения семпла на основе параметров."""
    param_str = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    safe_name = f"{method}_{param_str}".replace(".", "p")
    return SAMPLES_DIR / f"{safe_name}.pkl"


def _load_cached_sample(filepath: Path) -> Optional[nx.Graph]:
    """Загружает семпл из кеша, если существует."""
    if filepath.exists():
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None


def _save_sample(G: nx.Graph, filepath: Path) -> None:
    """Сохраняет семпл в кеш."""
    with open(filepath, 'wb') as f:
        pickle.dump(G, f)


# ----------------------------- 1. Случайный выбор узлов -----------------------------
def random_node_sampling(
    G: nx.Graph,
    fraction: Optional[float] = None,
    num_nodes: Optional[int] = None,
    seed: int = 42,
    use_cache: bool = True,
    graph_name: Optional[str] = None,
) -> nx.Graph:
    """
    Случайный выбор узлов и индуцированный подграф.

    Параметры
    ---------
    G : nx.Graph
        Исходный граф.
    fraction : float, optional
        Доля узлов для выбора (0 < fraction <= 1).
    num_nodes : int, optional
        Точное число узлов для выбора.
        Если указаны оба, приоритет у num_nodes.
    seed : int
        Сид для воспроизводимости.
    use_cache : bool
        Использовать кеширование.
    graph_name : str, optional
        Имя графа для включения в имя файла кеша.

    Возвращает
    -------
    nx.Graph
        Индуцированный подграф на выбранных узлах.
    """
    if fraction is None and num_nodes is None:
        raise ValueError("Укажите fraction или num_nodes")
    if num_nodes is None:
        num_nodes = max(1, int(G.number_of_nodes() * fraction))
    num_nodes = min(num_nodes, G.number_of_nodes())

    params = {
        'method': 'random_node',
        'num_nodes': num_nodes,
        'seed': seed,
    }
    if graph_name:
        params['graph'] = graph_name
    filepath = _get_sample_path("random_node", params)

    if use_cache:
        cached = _load_cached_sample(filepath)
        if cached is not None:
            print(f"📂 Загружен кешированный семпл: {filepath.name}")
            return cached

    random.seed(seed)
    nodes = list(G.nodes())
    sampled_nodes = random.sample(nodes, num_nodes)
    sample = G.subgraph(sampled_nodes).copy()

    if use_cache:
        _save_sample(sample, filepath)
        print(f"💾 Семпл сохранён: {filepath.name}")

    return sample


# ----------------------------- 2. Случайный выбор рёбер -----------------------------
def random_edge_sampling(
    G: nx.Graph,
    fraction: Optional[float] = None,
    num_edges: Optional[int] = None,
    seed: int = 42,
    use_cache: bool = True,
    graph_name: Optional[str] = None,
) -> nx.Graph:
    """
    Случайный выбор рёбер и индуцированный подграф на их вершинах.

    Параметры
    ---------
    G : nx.Graph
        Исходный граф.
    fraction : float, optional
        Доля рёбер для выбора.
    num_edges : int, optional
        Точное число рёбер.
    seed : int
        Сид.
    use_cache : bool
        Кеширование.
    graph_name : str, optional
        Имя графа.

    Возвращает
    -------
    nx.Graph
        Подграф, содержащий выбранные рёбра и инцидентные им вершины.
    """
    if fraction is None and num_edges is None:
        raise ValueError("Укажите fraction или num_edges")
    if num_edges is None:
        num_edges = max(1, int(G.number_of_edges() * fraction))
    num_edges = min(num_edges, G.number_of_edges())

    params = {
        'method': 'random_edge',
        'num_edges': num_edges,
        'seed': seed,
    }
    if graph_name:
        params['graph'] = graph_name
    filepath = _get_sample_path("random_edge", params)

    if use_cache:
        cached = _load_cached_sample(filepath)
        if cached is not None:
            print(f"📂 Загружен кешированный семпл: {filepath.name}")
            return cached

    random.seed(seed)
    edges = list(G.edges())
    sampled_edges = random.sample(edges, num_edges)
    sample = nx.Graph()
    sample.add_edges_from(sampled_edges)
    # Добавляем изолированные вершины? Обычно не нужно, но для честности можно:
    # nodes_in_edges = set([u for e in sampled_edges for u in e[:2]])
    # sample.add_nodes_from(nodes_in_edges)

    if use_cache:
        _save_sample(sample, filepath)
        print(f"💾 Семпл сохранён: {filepath.name}")

    return sample


# ----------------------------- 3. Snowball (BFS) семплирование ---------------------
def snowball_sampling(
    G: nx.Graph,
    fraction: Optional[float] = None,
    num_nodes: Optional[int] = None,
    start_node: Optional[int] = None,
    seed: int = 42,
    use_cache: bool = True,
    graph_name: Optional[str] = None,
) -> nx.Graph:
    """
    Семплирование снежным комом (BFS из случайного узла до достижения нужного числа вершин).

    Параметры
    ---------
    G : nx.Graph
        Исходный граф.
    fraction : float, optional
        Доля узлов для семпла.
    num_nodes : int, optional
        Точное число узлов.
    start_node : int, optional
        Стартовый узел. Если None, выбирается случайно.
    seed : int
        Сид.
    use_cache : bool
        Кеширование.
    graph_name : str, optional
        Имя графа.

    Возвращает
    -------
    nx.Graph
        Подграф, построенный BFS до нужного размера.
    """
    if fraction is None and num_nodes is None:
        raise ValueError("Укажите fraction или num_nodes")
    if num_nodes is None:
        num_nodes = max(1, int(G.number_of_nodes() * fraction))
    num_nodes = min(num_nodes, G.number_of_nodes())

    random.seed(seed)
    if start_node is None:
        start_node = random.choice(list(G.nodes()))
    else:
        if start_node not in G:
            raise ValueError(f"Узел {start_node} не найден в графе")

    params = {
        'method': 'snowball',
        'num_nodes': num_nodes,
        'start_node': start_node,
        'seed': seed,
    }
    if graph_name:
        params['graph'] = graph_name
    filepath = _get_sample_path("snowball", params)

    if use_cache:
        cached = _load_cached_sample(filepath)
        if cached is not None:
            print(f"📂 Загружен кешированный семпл: {filepath.name}")
            return cached

    visited = set()
    queue = [start_node]
    while queue and len(visited) < num_nodes:
        u = queue.pop(0)
        if u not in visited:
            visited.add(u)
            neighbors = list(G.neighbors(u))
            random.shuffle(neighbors)
            for v in neighbors:
                if v not in visited and len(visited) < num_nodes:
                    queue.append(v)

    sample = G.subgraph(visited).copy()
    if use_cache:
        _save_sample(sample, filepath)
        print(f"💾 Семпл сохранён: {filepath.name}")

    return sample


# ----------------------------- 4. Forest Fire семплирование -------------------------
def forest_fire_sampling(
    G: nx.Graph,
    fraction: Optional[float] = None,
    num_nodes: Optional[int] = None,
    p: float = 0.3,
    start_node: Optional[int] = None,
    seed: int = 42,
    use_cache: bool = True,
    graph_name: Optional[str] = None,
) -> nx.Graph:
    """
    Forest Fire семплирование (распространение "огня" с вероятностью p).

    Параметры
    ---------
    G : nx.Graph
        Исходный граф.
    fraction : float, optional
        Доля узлов.
    num_nodes : int, optional
        Точное число узлов.
    p : float
        Вероятность "поджечь" соседа.
    start_node : int, optional
        Стартовый узел (если None, случайный).
    seed : int
        Сид.
    use_cache : bool
        Кеширование.
    graph_name : str, optional
        Имя графа.

    Возвращает
    -------
    nx.Graph
        Подграф, построенный алгоритмом Forest Fire.
    """
    if fraction is None and num_nodes is None:
        raise ValueError("Укажите fraction или num_nodes")
    if num_nodes is None:
        num_nodes = max(1, int(G.number_of_nodes() * fraction))
    num_nodes = min(num_nodes, G.number_of_nodes())

    random.seed(seed)
    np.random.seed(seed)

    if start_node is None:
        start_node = random.choice(list(G.nodes()))
    else:
        if start_node not in G:
            raise ValueError(f"Узел {start_node} не найден в графе")

    params = {
        'method': 'forest_fire',
        'num_nodes': num_nodes,
        'p': p,
        'start_node': start_node,
        'seed': seed,
    }
    if graph_name:
        params['graph'] = graph_name
    filepath = _get_sample_path("forest_fire", params)

    if use_cache:
        cached = _load_cached_sample(filepath)
        if cached is not None:
            print(f"📂 Загружен кешированный семпл: {filepath.name}")
            return cached

    visited = set()
    active = [start_node]
    visited.add(start_node)

    while active and len(visited) < num_nodes:
        u = active.pop(0)
        neighbors = list(G.neighbors(u))
        random.shuffle(neighbors)
        for v in neighbors:
            if v not in visited:
                visited.add(v)
                active.append(v)
                # Геометрическое распределение числа "поджигаемых" соседей
                # Упростим: с вероятностью p добавляем следующего, иначе прекращаем
                if np.random.random() > p:
                    break
                if len(visited) >= num_nodes:
                    break
        if len(visited) >= num_nodes:
            break

    sample = G.subgraph(visited).copy()
    if use_cache:
        _save_sample(sample, filepath)
        print(f"💾 Семпл сохранён: {filepath.name}")

    return sample


# ----------------------------- 5. Random Walk семплирование -------------------------
def random_walk_sampling(
    G: nx.Graph,
    fraction: Optional[float] = None,
    num_nodes: Optional[int] = None,
    start_node: Optional[int] = None,
    seed: int = 42,
    use_cache: bool = True,
    graph_name: Optional[str] = None,
) -> nx.Graph:
    """
    Семплирование на основе случайного блуждания без возврата (до набора нужного числа уникальных вершин).

    Параметры
    ---------
    G : nx.Graph
        Исходный граф.
    fraction : float, optional
        Доля узлов.
    num_nodes : int, optional
        Точное число узлов.
    start_node : int, optional
        Стартовый узел.
    seed : int
        Сид.
    use_cache : bool
        Кеширование.
    graph_name : str, optional
        Имя графа.

    Возвращает
    -------
    nx.Graph
        Индуцированный подграф на посещённых узлах.
    """
    if fraction is None and num_nodes is None:
        raise ValueError("Укажите fraction или num_nodes")
    if num_nodes is None:
        num_nodes = max(1, int(G.number_of_nodes() * fraction))
    num_nodes = min(num_nodes, G.number_of_nodes())

    random.seed(seed)
    if start_node is None:
        start_node = random.choice(list(G.nodes()))
    else:
        if start_node not in G:
            raise ValueError(f"Узел {start_node} не найден в графе")

    params = {
        'method': 'random_walk',
        'num_nodes': num_nodes,
        'start_node': start_node,
        'seed': seed,
    }
    if graph_name:
        params['graph'] = graph_name
    filepath = _get_sample_path("random_walk", params)

    if use_cache:
        cached = _load_cached_sample(filepath)
        if cached is not None:
            print(f"📂 Загружен кешированный семпл: {filepath.name}")
            return cached

    visited = set()
    current = start_node
    visited.add(current)

    while len(visited) < num_nodes:
        neighbors = list(G.neighbors(current))
        if not neighbors:
            # Застряли – перепрыгиваем на случайный посещённый узел
            current = random.choice(list(visited))
            continue
        current = random.choice(neighbors)
        visited.add(current)

    sample = G.subgraph(visited).copy()
    if use_cache:
        _save_sample(sample, filepath)
        print(f"💾 Семпл сохранён: {filepath.name}")

    return sample


# ----------------------------- 6. Metropolis-Hastings Random Walk ------------------
def mhrw_sampling(
    G: nx.Graph,
    fraction: Optional[float] = None,
    num_nodes: Optional[int] = None,
    start_node: Optional[int] = None,
    seed: int = 42,
    use_cache: bool = True,
    graph_name: Optional[str] = None,
) -> nx.Graph:
    """
    Семплирование Metropolis-Hastings Random Walk (стремится к равномерному распределению по узлам).

    Параметры
    ---------
    G : nx.Graph
        Исходный граф.
    fraction : float, optional
        Доля узлов.
    num_nodes : int, optional
        Точное число узлов.
    start_node : int, optional
        Стартовый узел.
    seed : int
        Сид.
    use_cache : bool
        Кеширование.
    graph_name : str, optional
        Имя графа.

    Возвращает
    -------
    nx.Graph
        Индуцированный подграф на посещённых узлах.
    """
    if fraction is None and num_nodes is None:
        raise ValueError("Укажите fraction или num_nodes")
    if num_nodes is None:
        num_nodes = max(1, int(G.number_of_nodes() * fraction))
    num_nodes = min(num_nodes, G.number_of_nodes())

    random.seed(seed)
    if start_node is None:
        start_node = random.choice(list(G.nodes()))
    else:
        if start_node not in G:
            raise ValueError(f"Узел {start_node} не найден в графе")

    params = {
        'method': 'mhrw',
        'num_nodes': num_nodes,
        'start_node': start_node,
        'seed': seed,
    }
    if graph_name:
        params['graph'] = graph_name
    filepath = _get_sample_path("mhrw", params)

    if use_cache:
        cached = _load_cached_sample(filepath)
        if cached is not None:
            print(f"📂 Загружен кешированный семпл: {filepath.name}")
            return cached

    visited = set()
    current = start_node
    visited.add(current)

    while len(visited) < num_nodes:
        neighbors = list(G.neighbors(current))
        if not neighbors:
            current = random.choice(list(visited))
            continue
        candidate = random.choice(neighbors)
        # Степени текущего узла и кандидата
        deg_current = G.degree(current)
        deg_candidate = G.degree(candidate)
        # Вероятность перехода по Metropolis-Hastings
        acceptance = min(1.0, deg_current / deg_candidate)
        if random.random() <= acceptance:
            current = candidate
            visited.add(current)
        # Если отвергли, остаёмся на месте, но узел уже в visited

    sample = G.subgraph(visited).copy()
    if use_cache:
        _save_sample(sample, filepath)
        print(f"💾 Семпл сохранён: {filepath.name}")

    return sample

# ----------------------------- 7. Degree-Biased Sampling -----------------
def degree_biased_sampling(
    G: nx.Graph,
    fraction: Optional[float] = None,
    num_nodes: Optional[int] = None,
    edge_prob_scale: float = 1.0,
    seed: int = 42,
    use_cache: bool = True,
    graph_name: Optional[str] = None,
) -> nx.Graph:
    """
    Degree-Biased Sampling: выбор топ-узлов по степени и вероятностное добавление рёбер.

    Алгоритм:
    1. Выбираются top-k узлов с наибольшей степенью.
    2. Для каждого потенциального ребра между выбранными узлами:
       вероятность добавления пропорциональна сумме степеней концов:
       P(u,v) = min(1, edge_prob_scale * (deg(u) + deg(v)) / (2 * max_deg))

    Параметры
    ---------
    G : nx.Graph
        Исходный граф.
    fraction : float, optional
        Доля узлов для выбора.
    num_nodes : int, optional
        Точное число узлов для выбора.
    edge_prob_scale : float
        Множитель вероятности добавления ребра (по умолчанию 1.0).
    seed : int
        Сид для воспроизводимости.
    use_cache : bool
        Использовать кеширование.
    graph_name : str, optional
        Имя графа для включения в имя файла кеша.

    Возвращает
    -------
    nx.Graph
        Семпл, построенный по Degree-Biased алгоритму.
    """
    if fraction is None and num_nodes is None:
        raise ValueError("Укажите fraction или num_nodes")
    if num_nodes is None:
        num_nodes = max(1, int(G.number_of_nodes() * fraction))
    num_nodes = min(num_nodes, G.number_of_nodes())

    params = {
        'method': 'degree_biased',
        'num_nodes': num_nodes,
        'edge_prob_scale': edge_prob_scale,
        'seed': seed,
    }
    if graph_name:
        params['graph'] = graph_name
    filepath = _get_sample_path("degree_biased", params)

    if use_cache:
        cached = _load_cached_sample(filepath)
        if cached is not None:
            print(f"📂 Загружен кешированный Degree-Biased семпл: {filepath.name}")
            return cached

    random.seed(seed)
    np.random.seed(seed)

    # Шаг 1: Выбор топ-k узлов по степени
    nodes_by_degree = sorted(dict(G.degree()).items(), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in nodes_by_degree[:num_nodes]]
    top_set = set(top_nodes)

    # Шаг 2: Построение семпла с вероятностным добавлением рёбер
    sample = nx.Graph()
    sample.add_nodes_from(top_nodes)

    max_deg = max(dict(G.degree()).values())
    for i, u in enumerate(top_nodes):
        for v in top_nodes[i+1:]:
            if G.has_edge(u, v):
                deg_sum = G.degree(u) + G.degree(v)
                prob = edge_prob_scale * deg_sum / (2 * max_deg)
                if random.random() < min(1.0, prob):
                    sample.add_edge(u, v)

    if use_cache:
        _save_sample(sample, filepath)
        print(f"💾 Degree-Biased семпл сохранён: {filepath.name} (вершин: {sample.number_of_nodes()}, рёбер: {sample.number_of_edges()})")

    return sample


# ----------------------------- Универсальная обёртка --------------------------------
def sample_graph(
    G: nx.Graph,
    method: str = 'random_node',
    **params
) -> nx.Graph:
    """
    Универсальная функция для семплирования графа.

    Параметры
    ---------
    G : nx.Graph
        Исходный граф.
    method : {'random_node', 'random_edge', 'snowball', 'forest_fire', 'random_walk', 'mhrw'}
        Метод семплирования.
    **params : параметры соответствующего метода (fraction/num_nodes, seed, и т.д.)

    Возвращает
    -------
    nx.Graph
        Семпл.
    """
    methods = {
        'random_node': random_node_sampling,
        'random_edge': random_edge_sampling,
        'snowball': snowball_sampling,
        'forest_fire': forest_fire_sampling,
        'random_walk': random_walk_sampling,
        'mhrw': mhrw_sampling,
        'degree_biased': degree_biased_sampling,
    }
    if method not in methods:
        raise ValueError(f"Неизвестный метод: {method}. Доступны: {list(methods.keys())}")
    return methods[method](G, **params)