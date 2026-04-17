# src/features/graph_metrics.py
"""
Модуль для расчёта структурных метрик графа социальной сети.

Поддерживаемые группы метрик:
- basic: число вершин, рёбер, плотность, средняя степень
- degree: распределение степеней, моменты, power-law fit
- clustering: средний кластерный коэффициент, транзитивность
- paths: диаметр, средняя длина пути (приближённо для больших графов)
- centrality: degree, betweenness, closeness, eigenvector, pagerank, ksi
- spectral: алгебраическая связность (второе собственное значение лапласиана)
- communities: модулярность (требует предварительного разбиения)
"""

import numpy as np
import networkx as nx
import random
from typing import Dict, List, Optional, Union, Any, Tuple
from scipy import stats

# ----------------------------- Вспомогательные функции -----------------------------

def _ensure_undirected(G: nx.Graph) -> nx.Graph:
    """Приводит граф к неориентированному для метрик, требующих этого."""
    if G.is_directed():
        return G.to_undirected()
    return G

def _compute_ksi_centrality(G: nx.Graph) -> Dict[str, Union[np.ndarray, float]]:
    """
    Вычисляет ksi-центральность и нормализованную ksi-центральность
    согласно статье Tuzhilin (2025).
    """
    G = _ensure_undirected(G)
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    
    A2 = np.dot(A, A)
    J = np.ones((n, n))
    I = np.eye(n)
    A_bar = J - I - A  # матрица "не-соседей"
    
    numerator = np.diag(np.dot(A2, A_bar))
    denominator = np.diag(A2)
    
    # ksi-центральность
    with np.errstate(divide='ignore', invalid='ignore'):
        ksi = np.where(denominator > 0, numerator / denominator, 1.0)
    
    # нормализованная ksi-центральность
    degrees = np.sum(A, axis=1).astype(int)
    norm_ksi = np.zeros(n)
    for i in range(n):
        d_i = degrees[i]
        if d_i == 0:
            norm_ksi[i] = 1.0 / n
        else:
            norm_ksi[i] = numerator[i] / (d_i * (n - d_i))
    
    return {
        "ksi_centrality": ksi,
        "norm_ksi_centrality": norm_ksi,
        "avg_ksi": float(np.mean(ksi)),
        "avg_norm_ksi": float(np.mean(norm_ksi))
    }

def _powerlaw_fit(degrees: np.ndarray) -> Dict[str, float]:
    """Оценивает параметр степенного распределения методом максимального правдоподобия."""
    # Убираем нулевые степени
    data = degrees[degrees > 0]
    if len(data) < 10:
        return {"alpha": np.nan, "xmin": np.nan}
    try:
        # Простая оценка по Клаузу-Шализи-Ньюману
        xmin = min(data)
        n = len(data)
        alpha = 1 + n / np.sum(np.log(data / xmin))
        return {"alpha": alpha, "xmin": xmin}
    except:
        return {"alpha": np.nan, "xmin": np.nan}
    
def approx_diameter(G: nx.Graph, trials: int = 20, seed: int = 42) -> Optional[int]:
    """
    Оценка диаметра графа через BFS из trials случайных вершин.
    
    Параметры
    ---------
    G : nx.Graph
        Граф (должен быть связным).
    trials : int
        Число случайных стартовых вершин для BFS.
    seed : int
        Сид для воспроизводимости.
    
    Возвращает
    -------
    int или None
        Максимальный эксцентриситет среди выбранных вершин.
        Если граф пуст или несвязен, возвращает None.
    """
    if G.number_of_nodes() == 0:
        return None
    
    if not nx.is_connected(G):
        return None
    
    random.seed(seed)
    nodes = list(G.nodes())
    
    # Выбираем trials случайных вершин (или все, если trials > n)
    n_trials = min(trials, len(nodes))
    starts = random.sample(nodes, n_trials)
    
    max_ecc = 0
    for v in starts:
        try:
            # eccentricity возвращает максимальное расстояние от v до любой другой вершины
            ecc = nx.eccentricity(G, v)
            if ecc > max_ecc:
                max_ecc = ecc
        except:
            continue
    
    return max_ecc if max_ecc > 0 else None


def approx_average_shortest_path(
    G: nx.Graph, 
    num_pairs: int = 500, 
    seed: int = 42
) -> Optional[float]:
    """
    Оценка средней длины кратчайшего пути через сэмплирование случайных пар вершин.
    
    Параметры
    ---------
    G : nx.Graph
        Граф (должен быть связным).
    num_pairs : int
        Число случайных пар для оценки.
    seed : int
        Сид для воспроизводимости.
    
    Возвращает
    -------
    float или None
        Среднее расстояние между num_pairs случайными парами.
        Если граф несвязен или нет достижимых пар, возвращает None.
    """
    if G.number_of_nodes() < 2:
        return 0.0
    
    if not nx.is_connected(G):
        return None
    
    random.seed(seed)
    nodes = list(G.nodes())
    n = len(nodes)
    
    total_dist = 0
    valid_pairs = 0
    
    for _ in range(num_pairs):
        # Выбираем две случайные различные вершины
        u, v = random.sample(nodes, 2)
        try:
            dist = nx.shortest_path_length(G, source=u, target=v)
            total_dist += dist
            valid_pairs += 1
        except nx.NetworkXNoPath:
            # Не должно происходить в связном графе, но на всякий случай
            continue
    
    if valid_pairs == 0:
        return None
    
    return total_dist / valid_pairs

# ----------------------------- Основная функция ------------------------------------

def compute_graph_metrics(
    G: nx.Graph,
    metrics: Optional[List[str]] = None,
    large_graph: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Вычисляет набор метрик для заданного графа.

    Параметры
    ---------
    G : networkx.Graph
        Входной граф.
    metrics : list of str, optional
        Список групп метрик для вычисления. Доступны:
        'basic', 'degree', 'clustering', 'paths', 'centrality', 'spectral', 'communities'.
        Если None, вычисляются все, кроме 'communities' (требует разбиения).
    large_graph : bool, default False
        Если True, для ресурсоёмких метрик используются приближения или они пропускаются.
    **kwargs
        Дополнительные параметры, например, k для оценки диаметра.

    Возвращает
    -------
    dict
        Словарь с вычисленными метриками.
    """
    # Если метрики не указаны, берём все базовые (без сообществ)
    all_groups = ['basic', 'degree', 'clustering', 'paths', 'centrality', 'spectral']
    if metrics is None:
        metrics = all_groups
    else:
        # Проверка корректности
        for m in metrics:
            if m not in all_groups + ['communities']:
                raise ValueError(f"Неизвестная группа метрик: {m}")
    
    results = {}
    G_undirected = _ensure_undirected(G)
    n = G_undirected.number_of_nodes()
    m = G_undirected.number_of_edges()
    const_ops = 20000000
    base_k = int(const_ops // (n + m))
    
    # 1. Базовые метрики
    if 'basic' in metrics:
        results['num_nodes'] = n
        results['num_edges'] = m
        results['density'] = nx.density(G_undirected)
        results['avg_degree'] = (2 * m) / n if n > 0 else 0.0
        results['is_directed'] = G.is_directed()
        # Компоненты связности
        if G.is_directed():
            results['num_weakly_components'] = nx.number_weakly_connected_components(G)
            results['num_strongly_components'] = nx.number_strongly_connected_components(G)
        else:
            results['num_connected_components'] = nx.number_connected_components(G_undirected)
    
    # 2. Степенные метрики
    if 'degree' in metrics:
        degrees = np.array([d for _, d in G_undirected.degree()])
        results['degree_min'] = int(np.min(degrees))
        results['degree_max'] = int(np.max(degrees))
        results['degree_mean'] = float(np.mean(degrees))
        results['degree_std'] = float(np.std(degrees))
        results['degree_skewness'] = float(stats.skew(degrees))
        results['degree_kurtosis'] = float(stats.kurtosis(degrees))
        # Power-law fit
        results['powerlaw_alpha'], results['powerlaw_xmin'] = _powerlaw_fit(degrees).values()
    
    # 3. Кластерные метрики
    if 'clustering' in metrics:
        if large_graph or n > 50000:
            # Для очень больших графов оцениваем по выборке
            results['avg_clustering'] = nx.average_clustering(G_undirected, count_zeros=False)
        else:
            results['avg_clustering'] = nx.average_clustering(G_undirected)
        results['transitivity'] = nx.transitivity(G_undirected)
    
    # 4. Путевые метрики
    if 'paths' in metrics:
        # Проверяем связность
        if nx.is_connected(G_undirected):
            if large_graph or n > 1000:
                # Приближённый диаметр
                trials = kwargs.get('diameter_trials', 20)
                results['diameter_approx'] = approx_diameter(G_undirected, trials=base_k, seed=42)
                
                # Приближённая средняя длина пути
                num_pairs = kwargs.get('path_pairs', 500)
                results['avg_shortest_path_approx'] = approx_average_shortest_path(
                    G_undirected, num_pairs=base_k, seed=42
                )
            else:
                # Точные методы для маленьких графов
                results['diameter'] = nx.diameter(G_undirected)
                results['avg_shortest_path'] = nx.average_shortest_path_length(G_undirected)
        else:
            # Для несвязного графа берём наибольшую компоненту
            if G.is_directed():
                largest_cc = G_undirected.subgraph(max(nx.weakly_connected_components(G), key=len))
            else:
                largest_cc = G_undirected.subgraph(max(nx.connected_components(G_undirected), key=len))
            
            n_lcc = largest_cc.number_of_nodes()
            if n_lcc > 1:
                if large_graph or n_lcc > 1000:
                    results['diameter_lcc_approx'] = approx_diameter(largest_cc, trials=base_k, seed=42)
                    results['avg_shortest_path_lcc_approx'] = approx_average_shortest_path(
                        largest_cc, num_pairs=base_k, seed=42
                    )
                else:
                    results['diameter_lcc'] = nx.diameter(largest_cc)
                    results['avg_shortest_path_lcc'] = nx.average_shortest_path_length(largest_cc)
            
            results['fraction_lcc'] = n_lcc / n
    
    # 5. Центральности
    if 'centrality' in metrics:
        # Degree centrality (уже есть в degrees)
        # Betweenness - ресурсоёмкая, для больших графов либо приближение, либо пропуск
        if not large_graph and n < 1000:
            try:
                bc = nx.betweenness_centrality(G_undirected, normalized=True)
                results['avg_betweenness'] = float(np.mean(list(bc.values())))
                results['max_betweenness'] = float(np.max(list(bc.values())))
            except:
                results['avg_betweenness'] = None
        else:
            # Приближение через k узлов
            k = kwargs.get('k', base_k)
            bc = nx.betweenness_centrality(G_undirected, k=k, normalized=True, seed=42)
            results['avg_betweenness_approx'] = float(np.mean(list(bc.values())))
        
        # Closeness centrality
        if not large_graph and n < 1000:
            try:
                cc = nx.closeness_centrality(G_undirected)
                results['avg_closeness'] = float(np.mean(list(cc.values())))
            except:
                results['avg_closeness'] = None
        else:
            results['avg_closeness'] = None  # слишком дорого
        
        # Eigenvector centrality (может не сойтись)
        try:
            ec = nx.eigenvector_centrality_numpy(G_undirected)
            results['avg_eigenvector'] = float(np.mean(list(ec.values())))
        except:
            results['avg_eigenvector'] = None
        
        # PageRank
        pr = nx.pagerank(G_undirected, alpha=0.85)
        results['avg_pagerank'] = float(np.mean(list(pr.values())))
        
        # Ksi-центральность (всегда считаем, она относительно быстрая)
        ksi_stats = _compute_ksi_centrality(G_undirected)
        results.update(ksi_stats)
    
    # 6. Спектральные метрики
    if 'spectral' in metrics:
        try:
            L = nx.laplacian_matrix(G_undirected).astype(float)
            # Для больших графов используем частичное вычисление собственных чисел
            if large_graph and n > 5000:
                from scipy.sparse.linalg import eigsh
                # Вычисляем два наименьших собственных значения
                w = eigsh(L, k=2, which='SM', return_eigenvectors=False)
                results['algebraic_connectivity'] = float(w[1])  # второе наименьшее
            else:
                w = np.linalg.eigvalsh(L.toarray())
                w.sort()
                results['algebraic_connectivity'] = float(w[1])  # второе наименьшее
        except Exception as e:
            results['algebraic_connectivity'] = None
    
    # 7. Сообщества (опционально, требует предварительного разбиения)
    if 'communities' in metrics:
        # Если передано разбиение на сообщества в kwargs['partition']
        partition = kwargs.get('partition', None)
        if partition is not None:
            results['modularity'] = nx.community.modularity(G_undirected, partition)
        else:
            # Быстрое жадное разбиение (может быть медленным)
            if not large_graph or n < 10000:
                communities = nx.community.greedy_modularity_communities(G_undirected)
                partition = [{node: i for i, comm in enumerate(communities) for node in comm}]
                results['modularity'] = nx.community.modularity(G_undirected, communities)
            else:
                results['modularity'] = None
    
    return results