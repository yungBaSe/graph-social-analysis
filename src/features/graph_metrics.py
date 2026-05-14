# src/features/graph_metrics.py
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm
from scipy import stats

from src.data.data_loader import get_dataset

# === PATH CONFIGURATION ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

COMMUNITY_SIZE_LIMIT = 200_000
SPECTRAL_SIZE_LIMIT = 50_000
CLUSTERING_SIZE_LIMIT = 50_000

def _get_cache_path(name: str) -> Path:
    return DATA_PROCESSED / f"{name}_metrics.pkl"


def _ensure_undirected(G: nx.Graph | nx.DiGraph) -> nx.Graph:
    return G.to_undirected() if isinstance(G, nx.DiGraph) else G


def _powerlaw_fit(degrees: np.ndarray) -> Dict[str, float]:
    data = degrees[degrees > 0]
    if len(data) < 10:
        return {"alpha": np.nan, "xmin": np.nan}
    try:
        xmin = float(data.min())
        n = len(data)
        alpha = 1 + n / np.sum(np.log(data / xmin))
        return {"alpha": float(alpha), "xmin": int(xmin)}
    except:
        return {"alpha": np.nan, "xmin": np.nan}


def approx_diameter(G: nx.Graph, seed: int = 42) -> Optional[int]:
    if G.number_of_nodes() < 2 or not nx.is_connected(G):
        return None
    random.seed(seed)
    nodes = list(G.nodes())
    start = random.choice(nodes)
    dist = nx.single_source_shortest_path_length(G, start)
    far_node = max(dist, key=dist.get)
    dist2 = nx.single_source_shortest_path_length(G, far_node)
    return int(max(dist2.values()))


# ------------------------------------------------------------
#  Метрики по группам
# ------------------------------------------------------------
def compute_basic_metrics(G: nx.Graph | nx.DiGraph) -> Dict[str, Any]:
    G_u = _ensure_undirected(G)
    results = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G_u),
        "avg_degree": 2 * G.number_of_edges() / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        "is_directed": isinstance(G, nx.DiGraph),
    }
    if isinstance(G, nx.DiGraph):
        results["num_weakly_components"] = nx.number_weakly_connected_components(G)
        results["num_strongly_components"] = nx.number_strongly_connected_components(G)
    else:
        results["num_connected_components"] = nx.number_connected_components(G_u)
    return results


def compute_degree_metrics(G: nx.Graph | nx.DiGraph) -> Dict[str, Any]:
    G_u = _ensure_undirected(G)
    degrees = np.array([d for _, d in G_u.degree()])
    return {
        "degree_min": int(np.min(degrees)),
        "degree_max": int(np.max(degrees)),
        "degree_mean": float(np.mean(degrees)),
        "degree_std": float(np.std(degrees)),
        "degree_skewness": float(stats.skew(degrees)),
        "degree_kurtosis": float(stats.kurtosis(degrees)),
        "power_law_alpha": _powerlaw_fit(degrees)["alpha"],
        "assortativity": float(nx.degree_assortativity_coefficient(G_u)),
    }


def compute_clustering_metrics(G: nx.Graph | nx.DiGraph) -> Dict[str, float]:
    G_u = _ensure_undirected(G)
    return {
        "avg_clustering": float(nx.average_clustering(G_u)),
        "transitivity": float(nx.transitivity(G_u)),
    }


def compute_path_metrics(G: nx.Graph | nx.DiGraph) -> Dict[str, Any]:
    G_u = _ensure_undirected(G)
    if nx.is_connected(G_u):
        diameter = approx_diameter(G_u)
    else:
        diameters = []
        for cc in nx.connected_components(G_u):
            if len(cc) < 2:
                continue
            subgraph = G_u.subgraph(cc).copy()
            diam = approx_diameter(subgraph)
            if diam is not None:
                diameters.append(diam)
        diameter = max(diameters) if diameters else None
    return {"diameter": diameter}


def compute_centrality_metrics(G: nx.Graph | nx.DiGraph) -> Dict[str, Any]:
    """Только PageRank (быстрый)."""
    G_u = _ensure_undirected(G)
    pagerank = nx.pagerank(G_u, alpha=0.85)
    return {
        "pagerank_top10": dict(sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10])
    }


def compute_spectral_metrics(G: nx.Graph | nx.DiGraph) -> Dict[str, Any]:
    """Алгебраическая связность (второе собственное значение лапласиана)."""
    G_u = _ensure_undirected(G)
    if G_u.number_of_nodes() < 2:
        return {"algebraic_connectivity": None}
    try:
        L = nx.laplacian_matrix(G_u).astype(float)
        # Используем быстрое разреженное вычисление двух наименьших собственных значений
        from scipy.sparse.linalg import eigsh
        w = eigsh(L, k=2, which='SM', return_eigenvectors=False)
        alg_conn = float(w[1])
        return {"algebraic_connectivity": alg_conn}
    except:
        return {"algebraic_connectivity": None}


def compute_community_metrics(G: nx.Graph | nx.DiGraph) -> Dict[str, Any]:
    G_u = _ensure_undirected(G)
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G_u)
        modularity = community_louvain.modularity(partition, G_u)
        return {
            "modularity": float(modularity),
            "n_communities": len(set(partition.values())),
        }
    except Exception:
        try:
            communities = nx.community.greedy_modularity_communities(G_u)
            modularity = nx.community.modularity(G_u, communities)
            return {
                "modularity": float(modularity),
                "n_communities": len(communities),
            }
        except Exception:
            return {"modularity": None, "n_communities": None}


# ------------------------------------------------------------
#  Главная функция
# ------------------------------------------------------------
def compute_graph_metrics(
    G: nx.Graph | nx.DiGraph,
    name: Optional[str] = None,
    metric_groups: Optional[List[str]] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    if name is None:
        use_cache = False
        cache_path = None
    else:
        use_cache = True
        cache_path = _get_cache_path(name)

    if use_cache and not force_recompute and cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    n_nodes = G.number_of_nodes()

    # Группы по умолчанию: всё, что быстро и полезно
    if metric_groups is None:
        metric_groups = ['basic', 'degree', 'paths', 'centrality']
        if n_nodes <= COMMUNITY_SIZE_LIMIT:
            metric_groups.append('community')
        if n_nodes <= SPECTRAL_SIZE_LIMIT:
            metric_groups.append('spectral')
        if n_nodes <= CLUSTERING_SIZE_LIMIT:
            metric_groups.append('clustering')

    metrics = {
        "name": name or "custom_graph",
        "n_nodes": n_nodes,
        "large_graph": n_nodes > 50000,
        "is_directed": isinstance(G, nx.DiGraph),
        "computed_at": pd.Timestamp.now().isoformat(),
    }

    group_funcs = {
        'basic': lambda: compute_basic_metrics(G),
        'degree': lambda: compute_degree_metrics(G),
        'clustering': lambda: compute_clustering_metrics(G),
        'paths': lambda: compute_path_metrics(G),
        'centrality': lambda: compute_centrality_metrics(G),
        'spectral': lambda: compute_spectral_metrics(G),
        'community': lambda: compute_community_metrics(G),
    }

    print(f"Computing metrics for graph ({n_nodes:,} nodes)...")
    for group in tqdm(metric_groups, desc="Metric groups"):
        if group in group_funcs:
            metrics.update(group_funcs[group]())

    if use_cache:
        with open(cache_path, 'wb') as f:
            pickle.dump(metrics, f)
        print(f"✅ Metrics cached as '{name}'.\n")
    else:
        print("✅ Metrics computed (no cache).\n")

    return metrics


def compute_dataset_metrics(
    name: str,
    metric_groups: Optional[List[str]] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    dataset = get_dataset(name, verbose=False)
    return compute_graph_metrics(
        dataset["graph"],
        name=name,
        metric_groups=metric_groups,
        force_recompute=force_recompute
    )


def load_metrics(name: str) -> Dict[str, Any]:
    cache_path = _get_cache_path(name)
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    raise FileNotFoundError(f"No cached metrics for {name}")

def list_cached_metrics() -> None:
    files = sorted(DATA_PROCESSED.glob("*_metrics.pkl"))
    print(f"Found {len(files)} cached metric files:")
    for f in files:
        print(f"  • {f.name}")