import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
import gc
import seaborn as sns
import numpy as np
import networkx as nx
from typing import Optional

from src.data.data_loader import get_dataset
from src.features.graph_metrics import compute_dataset_metrics
from src.visualization.utils import show_and_save, set_style

def plot_degree_distribution(
    G: nx.Graph | nx.DiGraph,
    name: Optional[str] = None,
    show: bool = True,
    save: bool = True,
) -> plt.Figure:
    """Исправленная версия без зависаний."""
    if name is None:
        name = "custom_graph"

    degrees = np.array([d for _, d in G.degree()])

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    sns.histplot(degrees, bins=80, ax=axs[0], color='steelblue', alpha=0.85)
    axs[0].set_title(f"Degree Distribution — {name.upper()}")
    axs[0].set_xlabel("Degree")
    axs[0].set_ylabel("Count")

    # Безопасный log-scale
    pos_degrees = degrees[degrees > 0]
    if len(pos_degrees) > 0:
        sns.histplot(pos_degrees, bins=80, log_scale=True, 
                     ax=axs[1], color='steelblue', alpha=0.85)
    axs[1].set_title(f"Degree Distribution (log-log) — {name.upper()}")
    axs[1].set_xlabel("Degree (log)")
    axs[1].set_ylabel("Count (log)")

    plt.tight_layout()
    plt.close(fig)

    show_and_save(fig, name, "degree_distribution", show=show, save=save)

    return fig


def plot_clustering_distribution(
    G: nx.Graph | nx.DiGraph,
    name: Optional[str] = None,
    show: bool = True,
    save: bool = True,
) -> plt.Figure:
    """Local clustering coefficient distribution."""
    if name is None:
        name = "custom_graph"

    G_u = G.to_undirected() if G.is_directed() else G
    clustering = list(nx.clustering(G_u).values())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(clustering, bins=50, kde=True, color='darkorange', ax=ax)
    ax.set_title(f"Local Clustering Coefficient Distribution — {name.upper()}")
    ax.set_xlabel("Clustering Coefficient")
    ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.close(fig)

    show_and_save(fig, name, "clustering_distribution", show=show, save=save)
    
    return fig


def plot_all_distributions(
    G: nx.Graph | nx.DiGraph,
    name: Optional[str] = None,
    show: bool = True,
    save: bool = True,
) -> None:
    """Generate all useful distribution plots."""
    if name is None:
        name = "custom_graph"
    print(f"Generating distribution plots for {name}...")

    plot_degree_distribution(G, name, show=show, save=save)
    plot_clustering_distribution(G, name, show=show, save=save)

    print(f"✅ All distribution plots for {name} completed.\n")