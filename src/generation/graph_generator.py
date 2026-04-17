# src/generation/graph_generator.py
"""
Модуль для генерации синтетических графов.

Поддерживаемые модели:
- erdos_renyi (G(n,p) и G(n,m))
- watts_strogatz (малый мир)
- barabasi_albert (безмасштабная)
- stochastic_block_model (SBM) — для сообществ
- bter (Block Two-Erdős–Rényi) — упрощённая реализация

Все генераторы возвращают networkx.Graph и умеют кешировать результат.
"""

import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import networkx as nx
import numpy as np

# ----------------------------- Конфигурация -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "processed" / "synthetic"
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------- Вспомогательные функции -----------------------------
def _get_graph_path(model_name: str, params: Dict) -> Path:
    """Генерирует путь для сохранения графа на основе параметров."""
    param_str = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    safe_name = f"{model_name}_{param_str}".replace(".", "p")
    return SYNTHETIC_DIR / f"{safe_name}.pkl"


def _load_cached_graph(filepath: Path) -> Optional[nx.Graph]:
    """Загружает граф из кеша, если существует."""
    if filepath.exists():
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None


def _save_graph(G: nx.Graph, filepath: Path) -> None:
    """Сохраняет граф в кеш."""
    with open(filepath, 'wb') as f:
        pickle.dump(G, f)


# ----------------------------- 1. Erdős–Rényi -----------------------------
def erdos_renyi(
    n: int,
    p: Optional[float] = None,
    m: Optional[int] = None,
    directed: bool = False,
    seed: int = 42,
    use_cache: bool = True,
    base_graph_name: Optional[str] = None,
) -> nx.Graph:
    # Параметры для имени файла
    params = {'n': n, 'p': p, 'm': m, 'directed': directed, 'seed': seed}
    if base_graph_name:
        params['base_graph'] = base_graph_name
    filepath = _get_graph_path("er", params)

    if use_cache:
        cached = _load_cached_graph(filepath)
        if cached is not None:
            print(f"📂 Загружен кешированный ER граф: {filepath.name}")
            return cached

    random.seed(seed)
    np.random.seed(seed)

    # Параметры ТОЛЬКО для NetworkX (без лишних ключей)
    nx_params = {'n': n, 'seed': seed, 'directed': directed}
    if p is not None:
        nx_params['p'] = p
        G = nx.erdos_renyi_graph(**nx_params)
    elif m is not None:
        nx_params['m'] = m
        G = nx.gnm_random_graph(**nx_params)
    else:
        raise ValueError("Укажите ровно один из параметров: p или m")

    if use_cache:
        _save_graph(G, filepath)
        print(f"💾 ER граф сохранён: {filepath.name}")

    return G


# ----------------------------- 2. Watts–Strogatz -----------------------------
def watts_strogatz(
    n: int,
    k: int,
    p: float,
    seed: int = 42,
    use_cache: bool = True,
    base_graph_name: Optional[str] = None,
) -> nx.Graph:
    params = {'n': n, 'k': k, 'p': p, 'seed': seed}
    if base_graph_name:
        params['base_graph'] = base_graph_name
    filepath = _get_graph_path("ws", params)

    if use_cache:
        cached = _load_cached_graph(filepath)
        if cached is not None:
            print(f"📂 Загружен кешированный WS граф: {filepath.name}")
            return cached

    # Только нужные параметры
    G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)

    if use_cache:
        _save_graph(G, filepath)
        print(f"💾 WS граф сохранён: {filepath.name}")

    return G


# ----------------------------- 3. Barabási–Albert -----------------------------
def barabasi_albert(
    n: int,
    m: int,
    seed: int = 42,
    use_cache: bool = True,
    base_graph_name: Optional[str] = None,
) -> nx.Graph:
    params = {'n': n, 'm': m, 'seed': seed}
    if base_graph_name:
        params['base_graph'] = base_graph_name
    filepath = _get_graph_path("ba", params)

    if use_cache:
        cached = _load_cached_graph(filepath)
        if cached is not None:
            print(f"📂 Загружен кешированный BA граф: {filepath.name}")
            return cached

    G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)

    if use_cache:
        _save_graph(G, filepath)
        print(f"💾 BA граф сохранён: {filepath.name}")

    return G


# ----------------------------- 4. Stochastic Block Model ---------------------
def stochastic_block_model(
    sizes: List[int],
    p_matrix: Union[List[List[float]], np.ndarray],
    seed: int = 42,
    use_cache: bool = True,
    base_graph_name: Optional[str] = None,
) -> nx.Graph:
    """
    Генерирует граф с сообществами (Stochastic Block Model).

    Параметры
    ---------
    sizes : list
        Размеры блоков (сообществ).
    p_matrix : 2D array
        Матрица вероятностей рёбер между блоками.
    seed : int
        Сид.
    use_cache : bool
        Загружать из кеша.
    base_graph_name : str, optional
        Имя исходного графа для включения в имя файла кеша.

    Возвращает
    -------
    networkx.Graph
    """
    params = {'sizes': str(sizes), 'p_matrix': str(p_matrix), 'seed': seed}
    if base_graph_name:
        params['base_graph'] = base_graph_name
    filepath = _get_graph_path("sbm", params)

    if use_cache:
        cached = _load_cached_graph(filepath)
        if cached is not None:
            print(f"📂 Загружен кешированный SBM граф: {filepath.name}")
            return cached

    G = nx.stochastic_block_model(sizes, p_matrix, seed=seed)

    if use_cache:
        _save_graph(G, filepath)
        print(f"💾 SBM граф сохранён: {filepath.name}")

    return G


# ----------------------------- 5. BTER (Block Two-Erdős–Rényi) ---------------
def bter_graph(
    n: int,
    degree_distribution: Optional[np.ndarray] = None,
    clustering_coefficient: float = 0.1,
    seed: int = 42,
    use_cache: bool = True,
    base_graph_name: Optional[str] = None,
) -> nx.Graph:
    """
    Генерирует граф по упрощённой модели BTER (Block Two-Erdős–Rényi).

    Параметры
    ---------
    n : int
        Число вершин.
    degree_distribution : array, optional
        Желаемое распределение степеней. Если None, генерируется степенное.
    clustering_coefficient : float
        Целевой средний кластерный коэффициент.
    seed : int
        Сид.
    use_cache : bool
        Загружать из кеша.
    base_graph_name : str, optional
        Имя исходного графа для включения в имя файла кеша.

    Возвращает
    -------
    networkx.Graph
    """
    params = {'n': n, 'clustering': clustering_coefficient, 'seed': seed}
    if base_graph_name:
        params['base_graph'] = base_graph_name
    filepath = _get_graph_path("bter", params)

    if use_cache:
        cached = _load_cached_graph(filepath)
        if cached is not None:
            print(f"📂 Загружен кешированный BTER граф: {filepath.name}")
            return cached

    np.random.seed(seed)

    if degree_distribution is None:
        alpha = 2.5
        degrees = np.random.zipf(alpha, n)
        degrees = np.clip(degrees, 1, n - 1)
    else:
        degrees = degree_distribution

    sorted_idx = np.argsort(degrees)[::-1]
    block_size = max(1, n // 50)
    blocks = [sorted_idx[i : i + block_size] for i in range(0, n, block_size)]

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for block in blocks:
        if len(block) > 1:
            p_in = clustering_coefficient * 2
            for i, u in enumerate(block):
                for v in block[i + 1 :]:
                    if np.random.random() < p_in:
                        G.add_edge(u, v)

    current_degrees = np.array([G.degree(i) for i in range(n)])
    deficit = degrees - current_degrees

    nodes_with_deficit = [i for i in range(n) if deficit[i] > 0]
    for u in nodes_with_deficit:
        candidates = [v for v in nodes_with_deficit if v != u and not G.has_edge(u, v)]
        if not candidates:
            continue
        probs = np.array([deficit[v] for v in candidates], dtype=float)
        probs /= probs.sum()
        target = np.random.choice(candidates, p=probs)
        G.add_edge(u, target)
        deficit[u] -= 1
        deficit[target] -= 1

    if use_cache:
        _save_graph(G, filepath)
        print(f"💾 BTER граф сохранён: {filepath.name}")

    return G


# ----------------------------- Универсальная обёртка ------------------------
def generate_graph(model: str, **params) -> nx.Graph:
    """
    Универсальная функция для генерации синтетического графа.

    Параметры
    ---------
    model : {'er', 'ws', 'ba', 'sbm', 'bter'}
        Модель генерации.
    **params : параметры соответствующей модели.

    Возвращает
    -------
    networkx.Graph
    """
    if model.lower() == 'er':
        return erdos_renyi(**params)
    elif model.lower() == 'ws':
        return watts_strogatz(**params)
    elif model.lower() == 'ba':
        return barabasi_albert(**params)
    elif model.lower() == 'sbm':
        return stochastic_block_model(**params)
    elif model.lower() == 'bter':
        return bter_graph(**params)
    else:
        raise ValueError(f"Неизвестная модель: {model}. Доступны: er, ws, ba, sbm, bter")


def generate_matching_graph(
    real_graph: nx.Graph,
    model: str = 'ba',
    base_graph_name: Optional[str] = None,
    **extra_params,
) -> nx.Graph:
    """
    Генерирует синтетический граф, сопоставимый по размеру с реальным.

    Параметры
    ---------
    real_graph : nx.Graph
        Реальный граф, чьи параметры (n, m) будут скопированы.
    model : str
        Модель генерации.
    base_graph_name : str, optional
        Имя исходного графа для включения в имя файла кеша.
    **extra_params : дополнительные параметры модели.

    Возвращает
    -------
    nx.Graph
    """
    n = real_graph.number_of_nodes()
    m = real_graph.number_of_edges()

    params = {'n': n, 'seed': extra_params.pop('seed', 42)}
    if base_graph_name:
        params['base_graph_name'] = base_graph_name

    if model == 'er':
        p = 2 * m / (n * (n - 1)) if n > 1 else 0
        params['p'] = p
    elif model == 'ws':
        avg_deg = int(2 * m / n)
        k = avg_deg if avg_deg % 2 == 0 else avg_deg + 1
        params['k'] = min(k, n - 1)
        params['p'] = extra_params.pop('p', 0.1)
    elif model == 'ba':
        params['m'] = max(1, int(m / n))
    elif model == 'bter':
        params['n'] = n
        if 'clustering_coefficient' not in extra_params:
            G_und = real_graph.to_undirected() if real_graph.is_directed() else real_graph
            extra_params['clustering_coefficient'] = nx.average_clustering(G_und)
    else:
        # Для sbm и других моделей просто передаём параметры
        pass

    params.update(extra_params)
    return generate_graph(model, **params)