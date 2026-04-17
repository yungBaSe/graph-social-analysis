# src/visualization/visualize.py
"""
Модуль для визуализации графов, метрик и эмбеддингов.

Возможности:
- Гистограммы распределений метрик (степени, кластеризация, ksi)
- Сравнение нескольких графов по набору метрик (радарные диаграммы, heatmap)
- Визуализация эго-сетей и хабов
- t-SNE / PCA / UMAP для эмбеддингов
- Сохранение графиков в figures/
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

# ----------------------------- Конфигурация -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Настройка стиля matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Цветовая палитра для разных графов
COLORS = {
    'facebook': '#3b5998',
    'arxiv': '#2ecc71',
    'lastfm': '#e74c3c',
    'real': '#3498db',
    'synthetic': '#e67e22',
    'barabasi': '#9b59b6',
    'watts': '#1abc9c',
    'erdos': '#95a5a6',
}

# ----------------------------- 1. РАСПРЕДЕЛЕНИЯ МЕТРИК -----------------------------

def plot_degree_distribution(
    G: nx.Graph,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    color: str = 'blue',
    log_scale: bool = True,
    save_path: Optional[Path] = None,
    **kwargs
) -> plt.Axes:
    """
    Строит гистограмму распределения степеней вершин.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    degrees = [d for _, d in G.degree()]
    
    if log_scale:
        bins = np.logspace(np.log10(max(1, min(degrees))), 
                          np.log10(max(degrees)), 50)
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        bins = 50
    
    ax.hist(degrees, bins=bins, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Степень вершины')
    ax.set_ylabel('Частота')
    ax.set_title(title or f'Распределение степеней (n={G.number_of_nodes():,})')
    
    if save_path:
        plt.savefig(FIGURES_DIR / save_path, bbox_inches='tight')
    
    return ax

def compare_degree_distributions(
    G1: nx.Graph,
    G2: nx.Graph,
    label1: str = "Граф 1",
    label2: str = "Граф 2",
    title: str = "Сравнение распределений степеней",
    log_scale: bool = True,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Path] = None,
    **kwargs
) -> plt.Axes:
    """
    Сравнивает распределения степеней двух графов.
    
    Параметры
    ---------
    G1, G2 : nx.Graph
        Сравниваемые графы.
    label1, label2 : str
        Подписи для легенды.
    title : str
        Заголовок графика.
    log_scale : bool
        Использовать логарифмический масштаб по обеим осям.
    ax : plt.Axes, optional
        Оси для рисования.
    save_path : Path, optional
        Путь для сохранения.
    
    Возвращает
    -------
    ax : plt.Axes
    """
    from scipy.stats import ks_2samp
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    degrees1 = np.array([d for _, d in G1.degree()])
    degrees2 = np.array([d for _, d in G2.degree()])
    
    # KS-тест
    ks_stat, p_value = ks_2samp(degrees1, degrees2)
    
    # Цвета
    color1 = kwargs.get('color1', '#3498db')
    color2 = kwargs.get('color2', '#e74c3c')
    
    # Бины
    if log_scale:
        min_deg = max(1, min(degrees1.min(), degrees2.min()))
        max_deg = max(degrees1.max(), degrees2.max())
        bins = np.logspace(np.log10(min_deg), np.log10(max_deg), 40)
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        bins = 40
    
    # Гистограммы
    ax.hist(degrees1, bins=bins, alpha=0.5, color=color1, density=True, label=label1)
    ax.hist(degrees2, bins=bins, alpha=0.5, color=color2, density=True, label=label2)
    
    # KDE для гладкости
    try:
        from scipy import stats
        if log_scale:
            x_range = np.logspace(np.log10(max(1, degrees1[degrees1>0].min())), 
                                  np.log10(degrees1.max()), 100)
        else:
            x_range = np.linspace(degrees1.min(), degrees1.max(), 100)
        kde1 = stats.gaussian_kde(degrees1)
        ax.plot(x_range, kde1(x_range), color=color1, linewidth=2)
        
        if log_scale:
            x_range = np.logspace(np.log10(max(1, degrees2[degrees2>0].min())), 
                                  np.log10(degrees2.max()), 100)
        else:
            x_range = np.linspace(degrees2.min(), degrees2.max(), 100)
        kde2 = stats.gaussian_kde(degrees2)
        ax.plot(x_range, kde2(x_range), color=color2, linewidth=2)
    except:
        pass
    
    ax.set_xlabel('Степень вершины')
    ax.set_ylabel('Плотность')
    ax.set_title(f"{title}\nKS-статистика = {ks_stat:.4f}, p-value = {p_value:.4e}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(FIGURES_DIR / save_path, bbox_inches='tight')
    
    return ax

def plot_centrality_distribution(
    centrality_values: np.ndarray,
    centrality_name: str,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    color: str = 'blue',
    bins: int = 50,
    log_x: bool = False,
    clip_percentile: Optional[float] = 99.5,
    save_path: Optional[Path] = None,
    **kwargs
) -> plt.Axes:
    """
    Строит гистограмму распределения значений центральности.
    
    Параметры
    ---------
    log_x : bool
        Использовать логарифмическую шкалу по X.
    clip_percentile : float or None
        Процентиль для обрезания выбросов (None = не обрезать).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    values = centrality_values[~np.isnan(centrality_values)]
    
    # Обрезаем выбросы
    if clip_percentile is not None and clip_percentile < 100:
        upper = np.percentile(values, clip_percentile)
        values = values[values <= upper]
    
    # Для log шкалы убираем нули
    if log_x:
        values = values[values > 0]
        if len(values) == 0:
            print(f"⚠️ Нет положительных значений для {centrality_name}, log шкала недоступна")
            log_x = False
    
    if log_x:
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), bins)
        ax.set_xscale('log')
        xlabel = f'{centrality_name} (log scale)'
    else:
        xlabel = centrality_name
    
    ax.hist(values, bins=bins, alpha=0.7, color=color, 
            edgecolor='black', linewidth=0.5, density=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Плотность')
    ax.set_title(title or f'Распределение {centrality_name}')
    
    if save_path:
        plt.savefig(FIGURES_DIR / save_path, bbox_inches='tight')
    
    return ax


def compare_distributions(
    distributions: Dict[str, np.ndarray],
    title: str = 'Сравнение распределений',
    xlabel: str = 'Значение',
    ylabel: str = 'Плотность',
    colors: Optional[Dict[str, str]] = None,
    log_x: bool = False,  # <-- новый параметр
    clip_percentile: Optional[float] = 99.0,  # <-- новый параметр
    save_path: Optional[Path] = None,
    **kwargs
) -> plt.Axes:
    """
    Сравнивает несколько распределений на одном графике.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (name, values) in enumerate(distributions.items()):
        # Очистка и обрезание
        values = values[~np.isnan(values)]
        if clip_percentile is not None:
            upper = np.percentile(values, clip_percentile)
            values = values[values <= upper]
        if log_x:
            values = values[values > 0]
        
        if len(values) < 2:
            continue
            
        color = colors.get(name, f'C{i}') if colors else f'C{i}'
        
        from scipy import stats
        try:
            kde = stats.gaussian_kde(values)
            if log_x:
                x_range = np.logspace(np.log10(values.min()), np.log10(values.max()), 200)
            else:
                x_range = np.linspace(values.min(), values.max(), 200)
            ax.plot(x_range, kde(x_range), label=name, color=color, linewidth=2)
        except:
            pass
        
        # Гистограмма
        if log_x:
            bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 30)
        else:
            bins = 30
        ax.hist(values, bins=bins, alpha=0.2, color=color, density=True)
    
    if log_x:
        ax.set_xscale('log')
        xlabel += ' (log scale)'
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        plt.savefig(FIGURES_DIR / save_path, bbox_inches='tight')
    
    return ax


# ----------------------------- 2. СРАВНЕНИЕ ГРАФОВ -----------------------------

def plot_radar_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metrics_to_show: Optional[List[str]] = None,
    title: str = 'Сравнение графов по метрикам',
    colors: Optional[Dict[str, str]] = None,
    save_path: Optional[Path] = None,
    **kwargs
) -> plt.Axes:
    """
    Строит радарную (паутинную) диаграмму для сравнения нескольких графов.
    
    Параметры
    ---------
    metrics_dict : dict
        {имя_графа: {метрика: значение, ...}}
    metrics_to_show : list, optional
        Список метрик для отображения. Если None — все, что есть в первом графе.
    """
    import matplotlib.pyplot as plt
    from math import pi
    
    if metrics_to_show is None:
        # Берём метрики из первого графа
        first_graph = list(metrics_dict.keys())[0]
        metrics_to_show = list(metrics_dict[first_graph].keys())
    
    # Нормализуем значения для радарной диаграммы (от 0 до 1)
    normalized = {}
    for metric in metrics_to_show:
        values = [metrics_dict[g].get(metric, 0) for g in metrics_dict]
        min_v, max_v = min(values), max(values)
        if max_v - min_v < 1e-10:
            normalized[metric] = {g: 0.5 for g in metrics_dict}
        else:
            normalized[metric] = {g: (metrics_dict[g].get(metric, 0) - min_v) / (max_v - min_v) 
                                  for g in metrics_dict}
    
    # Подготовка углов
    N = len(metrics_to_show)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # замыкаем круг
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    for i, (graph_name, _) in enumerate(metrics_dict.items()):
        values = [normalized[m][graph_name] for m in metrics_to_show]
        values += values[:1]
        color = colors.get(graph_name, f'C{i}') if colors else f'C{i}'
        ax.plot(angles, values, 'o-', linewidth=2, label=graph_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_show)
    ax.set_ylim(0, 1)
    ax.set_title(title, size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    if save_path:
        plt.savefig(FIGURES_DIR / save_path, bbox_inches='tight')
    
    return ax


def plot_metrics_heatmap(
    metrics_dict: Dict[str, Dict[str, float]],
    normalize: bool = True,
    title: str = 'Тепловая карта метрик',
    save_path: Optional[Path] = None,
    **kwargs
) -> plt.Axes:
    """
    Строит тепловую карту сравнения метрик для разных графов.
    """
    import seaborn as sns
    
    graph_names = list(metrics_dict.keys())
    all_metrics = set()
    for g in metrics_dict:
        all_metrics.update(metrics_dict[g].keys())
    metrics_list = sorted(list(all_metrics))
    
    # Формируем матрицу
    data = []
    for graph in graph_names:
        row = [metrics_dict[graph].get(m, np.nan) for m in metrics_list]
        data.append(row)
    
    data = np.array(data)
    
    if normalize:
        # Нормализуем по столбцам
        data = (data - np.nanmin(data, axis=0)) / (np.nanmax(data, axis=0) - np.nanmin(data, axis=0) + 1e-10)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(graph_names) * 1.5)))
    sns.heatmap(data, annot=True, fmt='.2f', xticklabels=metrics_list, 
                yticklabels=graph_names, cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Нормированное значение'})
    ax.set_title(title)
    ax.set_xlabel('Метрики')
    ax.set_ylabel('Графы')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(FIGURES_DIR / save_path, bbox_inches='tight')
    
    return ax


# ----------------------------- 3. ВИЗУАЛИЗАЦИЯ ГРАФА -----------------------------

def plot_ego_network(
    G: nx.Graph,
    center_node: Optional[int] = None,
    radius: int = 1,
    node_size_by: str = 'local_degree',  # changed default
    node_color_by: str = 'distance',     # changed default
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Path] = None,
    max_node_size: int = 300,
    min_node_size: int = 30,
    **kwargs
) -> plt.Axes:
    """
    Визуализирует эго-сеть вокруг заданного узла.
    
    Интерпретация:
    - Центральный узел (красный) — эго.
    - Большие узлы — важные в пределах эго-сети (много связей).
    - Цвет показывает расстояние от центра: синий = сам центр, голубой = соседи, серый = соседи соседей.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Выбираем центральный узел
    if center_node is None:
        center_node = max(dict(G.degree()).items(), key=lambda x: x[1])[0]
    
    # Извлекаем эго-сеть
    ego = nx.ego_graph(G, center_node, radius=radius)
    
    # --- Размеры узлов ---
    if node_size_by == 'local_degree':
        degrees = dict(ego.degree())
        # Нормализуем размеры в заданном диапазоне
        max_deg = max(degrees.values()) if degrees else 1
        min_deg = min(degrees.values()) if degrees else 1
        if max_deg == min_deg:
            sizes = {n: (min_node_size + max_node_size) / 2 for n in ego.nodes()}
        else:
            sizes = {n: min_node_size + (degrees[n] - min_deg) / (max_deg - min_deg) * (max_node_size - min_node_size) 
                     for n in ego.nodes()}
    elif node_size_by == 'degree':
        # Используем глобальную степень
        global_deg = dict(G.degree())
        max_deg = max(global_deg.values())
        sizes = {n: min_node_size + (global_deg[n] / max_deg) * (max_node_size - min_node_size) for n in ego.nodes()}
    else:
        sizes = {n: (min_node_size + max_node_size) / 2 for n in ego.nodes()}
    
    # --- Цвета узлов ---
    if node_color_by == 'distance':
        # Расстояние от центрального узла
        distances = nx.single_source_shortest_path_length(ego, center_node)
        max_dist = max(distances.values())
        cmap = plt.cm.coolwarm
        colors = [distances[n] for n in ego.nodes()]
    elif node_color_by == 'local_degree':
        degrees = dict(ego.degree())
        colors = [degrees[n] for n in ego.nodes()]
        cmap = plt.cm.viridis
    else:
        colors = 'lightblue'
        cmap = None
    
    # Позиции
    pos = nx.spring_layout(ego, seed=42, k=1.5, iterations=50)
    
    # Рисуем рёбра
    nx.draw_networkx_edges(ego, pos, alpha=0.4, edge_color='gray', width=0.8)
    
    # Рисуем узлы
    nodes = nx.draw_networkx_nodes(
        ego, pos,
        node_size=[sizes[n] for n in ego.nodes()],
        node_color=colors,
        cmap=cmap,
        alpha=0.9
    )
    
    # Выделяем центральный узел красной границей
    nx.draw_networkx_nodes(ego, pos, nodelist=[center_node], 
                           node_color='none', edgecolors='red', linewidths=2.5,
                           node_size=sizes[center_node] * 1.2)
    
    # Подписи: только центральный и, возможно, ключевые соседи (опционально)
    labels = {center_node: str(center_node)}
    if kwargs.get('label_neighbors', False):
        for neighbor in ego.neighbors(center_node):
            labels[neighbor] = str(neighbor)
    nx.draw_networkx_labels(ego, pos, labels, font_size=8)
    
    if cmap:
        cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
        cbar.set_label(node_color_by)
    
    ax.set_title(title or f'Эго-сеть узла {center_node} (радиус {radius})\nРазмер = локальная степень, цвет = расстояние')
    ax.axis('off')
    
    if save_path:
        plt.savefig(FIGURES_DIR / save_path, bbox_inches='tight', dpi=150)
    
    return ax


def plot_hubs(
    G: nx.Graph,
    top_k: Optional[int] = None,        # если None, берём топ 1%
    top_frac: float = 0.01,             # доля узлов, считаемых хабами (если top_k не задан)
    metric: str = 'degree',
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Path] = None,
    max_node_size: int = 200,
    min_node_size: int = 10,
    max_visible_nodes: int = 250,       # <-- новый параметр: макс. число отображаемых узлов
    **kwargs
) -> plt.Axes:
    """
    Визуализирует граф с выделением хабов (узлов с экстремальной центральностью).
    Для больших графов отображает только топ-N узлов по центральности.
    
    Параметры
    ---------
    top_k : int, optional
        Число хабов для выделения. Если None, используется top_frac.
    top_frac : float
        Доля узлов, выделяемых как хабы (по умолчанию 1%).
    metric : str
        Метрика центральности: 'degree', 'pagerank', 'betweenness', 'closeness', 'eigenvector'.
    max_visible_nodes : int
        Максимальное число узлов для отображения (остальные скрываются).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    n = G.number_of_nodes()
    if top_k is None:
        top_k = max(1, int(n * top_frac))
    
    # Вычисляем метрику
    if metric == 'degree':
        scores = dict(G.degree())
    elif metric == 'pagerank':
        scores = nx.pagerank(G, alpha=0.85)
    elif metric == 'betweenness':
        k = min(1000, n)
        scores = nx.betweenness_centrality(G, k=k, seed=42)
    elif metric == 'closeness':
        scores = nx.closeness_centrality(G)
    elif metric == 'eigenvector':
        try:
            scores = nx.eigenvector_centrality_numpy(G)
        except:
            scores = nx.eigenvector_centrality(G, max_iter=500)
    else:
        scores = dict(G.degree())
    
    # Сортируем узлы по убыванию метрики
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Определяем хабы (топ top_k)
    hub_nodes = [node for node, _ in sorted_nodes[:top_k]]
    
    # Определяем видимые узлы (хабы + остальные до max_visible_nodes)
    visible_nodes = [node for node, _ in sorted_nodes[:max_visible_nodes]]
    visible_hubs = [n for n in hub_nodes if n in visible_nodes]
    
    # Если граф слишком большой, создаём подграф только с видимыми узлами
    if n > max_visible_nodes:
        G_vis = G.subgraph(visible_nodes).copy()
        print(f"⚠️ Граф сокращён с {n} до {max_visible_nodes} узлов для визуализации")
    else:
        G_vis = G.copy()
        visible_hubs = hub_nodes
    
    # Пересчитываем scores только для видимых узлов
    vis_scores = {n: scores[n] for n in visible_nodes}
    max_score = max(vis_scores.values()) if vis_scores else 1
    min_score = min(vis_scores.values()) if vis_scores else 0
    
    def scale_size(score):
        if max_score == min_score:
            return (min_node_size + max_node_size) / 2
        return min_node_size + (score - min_score) / (max_score - min_score) * (max_node_size - min_node_size)
    
    node_sizes = [2 * scale_size(vis_scores.get(n, 0)) for n in G_vis.nodes()]
    node_colors = ['#ff4444' if n in visible_hubs else '#a0a0a0' for n in G_vis.nodes()]
    edge_colors = ['#ff0000' if (u in visible_hubs or v in visible_hubs) else "#11c450" 
                   for u, v in G_vis.edges()]
    edge_widths = [2 if (u in visible_hubs or v in visible_hubs) else 0.5 
                   for u, v in G_vis.edges()]
    
    # Layout только для видимого подграфа
    pos = nx.spring_layout(G_vis, seed=42, k=2.0, iterations=50)
    
    # Рисуем рёбра
    nx.draw_networkx_edges(G_vis, pos, edge_color=edge_colors, width=edge_widths, alpha=0.4)
    
    # Рисуем узлы
    nx.draw_networkx_nodes(G_vis, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    
    # Подписи только для хабов (среди видимых)
    labels = {n: str(n) for n in visible_hubs}
    nx.draw_networkx_labels(G_vis, pos, labels, font_size=9, font_weight='bold')
    
    # Заголовок
    if n > max_visible_nodes:
        ax.set_title(title or f'Топ-{len(visible_hubs)} хабов по {metric} (показаны топ-{max_visible_nodes} узлов)')
    else:
        ax.set_title(title or f'Топ-{len(visible_hubs)} хабов по {metric}')
    ax.axis('off')
    
    # Легенда
    import matplotlib.patches as mpatches
    hub_patch = mpatches.Patch(color='#ff4444', label=f'Хабы (топ {len(visible_hubs)})')
    other_patch = mpatches.Patch(color='#a0a0a0', label='Остальные узлы')
    ax.legend(handles=[hub_patch, other_patch], loc='upper right')
    
    if save_path:
        plt.savefig(FIGURES_DIR / save_path, bbox_inches='tight', dpi=150)
    
    return ax


# ----------------------------- 4. ВИЗУАЛИЗАЦИЯ ЭМБЕДДИНГОВ -----------------------------

def plot_embeddings_2d(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = 'tsne',
    title: str = 'Визуализация эмбеддингов',
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Path] = None,
    random_state: int = 42,
    **kwargs
) -> plt.Axes:
    """
    Визуализирует эмбеддинги в 2D с помощью t-SNE, PCA или UMAP.
    
    Параметры
    ---------
    embeddings : np.ndarray
        Матрица эмбеддингов (n_samples x dim).
    labels : np.ndarray, optional
        Метки для окраски точек (например, степени или сообщества).
    method : {'tsne', 'pca', 'umap'}
        Метод снижения размерности.
    **kwargs : дополнительные параметры для метода снижения размерности.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Снижение размерности
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2, 
            random_state=random_state,
            perplexity=kwargs.get('perplexity', 30),
            learning_rate=kwargs.get('learning_rate', 200),
            n_iter=kwargs.get('n_iter', 1000)
        )
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2,
                random_state=random_state,
                n_neighbors=kwargs.get('n_neighbors', 15),
                min_dist=kwargs.get('min_dist', 0.1)
            )
        except ImportError:
            raise ImportError("umap-learn не установлен. pip install umap-learn")
    else:
        raise ValueError(f"Неизвестный метод: {method}")
    
    print(f"🔄 Снижение размерности методом {method.upper()}...")
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Визуализация
    if labels is not None:
        scatter = ax.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=labels, cmap='viridis', alpha=0.6, s=10
        )
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                  alpha=0.6, s=10, c='steelblue')
    
    ax.set_title(title)
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    
    if save_path:
        plt.savefig(FIGURES_DIR / save_path, bbox_inches='tight')
    
    return ax


def compare_embeddings(
    real_emb: np.ndarray,
    synthetic_emb: np.ndarray,
    method: str = 'tsne',
    title: str = 'Сравнение эмбеддингов: реальный vs синтетический',
    save_path: Optional[Path] = None,
    **kwargs
) -> Tuple[plt.Axes, plt.Axes]:
    """
    Сравнивает эмбеддинги реального и синтетического графов на одном рисунке.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Объединяем для общего снижения размерности
    combined = np.vstack([real_emb, synthetic_emb])
    n_real = real_emb.shape[0]
    
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, 
                      perplexity=kwargs.get('perplexity', 30))
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Неизвестный метод: {method}")
    
    combined_2d = reducer.fit_transform(combined)
    real_2d = combined_2d[:n_real]
    synth_2d = combined_2d[n_real:]
    
    # Рисуем
    ax1.scatter(real_2d[:, 0], real_2d[:, 1], alpha=0.5, s=8, c='blue')
    ax1.set_title('Реальный граф')
    ax1.set_xlabel(f'{method.upper()} 1')
    ax1.set_ylabel(f'{method.upper()} 2')
    
    ax2.scatter(synth_2d[:, 0], synth_2d[:, 1], alpha=0.5, s=8, c='orange')
    ax2.set_title('Синтетический граф')
    ax2.set_xlabel(f'{method.upper()} 1')
    ax2.set_ylabel(f'{method.upper()} 2')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(FIGURES_DIR / save_path, bbox_inches='tight')
    
    return ax1, ax2

# ----------------------------- 6. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ -----------------------

def save_figure(fig: Optional[plt.Figure] = None, name: str = 'figure') -> None:
    """Сохраняет текущую фигуру в папку figures."""
    if fig is None:
        fig = plt.gcf()
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"💾 Рисунок сохранён: {path}")  