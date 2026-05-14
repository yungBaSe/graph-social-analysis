import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
import networkx as nx
import numpy as np
from typing import Optional
import community as community_louvain
from collections import defaultdict

from src.data.data_loader import get_dataset
from src.visualization.utils import show_and_save, set_style


def plot_degree_rank(
    G: nx.Graph | nx.DiGraph,
    name: Optional[str] = None,
    show: bool = True,
    save: bool = True,
) -> plt.Figure:
    """Log-log degree rank plot (Zipf's law)."""
    if name is None:
        name = "custom_graph"

    degrees = sorted([d for _, d in G.degree()], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ranks = np.arange(1, len(degrees) + 1)
    ax.loglog(ranks, degrees, marker='.', linestyle='none', alpha=0.75, color='steelblue')
    ax.set_xlabel("Rank (log)")
    ax.set_ylabel("Degree (log)")
    ax.set_title(f"Degree Rank Plot (Zipf's law) — {name.upper()}")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return fig


def plot_ego_network(
    G: nx.Graph | nx.DiGraph,
    name: Optional[str] = None,
    center_node: Optional[int] = None,
    radius: int = 1,
    show: bool = True,
    save: bool = True,
) -> plt.Figure:
    """Визуализация ego-сети с выделением связей между соседями центра."""
    if name is None:
        name = "custom_graph"

    if center_node is None:
        center_node = max(dict(G.degree()).items(), key=lambda x: x[1])[0]

    ego = nx.ego_graph(G, center_node, radius=radius)
    nodes_1hop = set(G.neighbors(center_node))
    nodes_2hop = set(ego.nodes()) - {center_node} - nodes_1hop

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(ego, seed=42, k=0.35, iterations=80)

    # Рёбра между 1-hop соседями
    edges_1hop_internal = [(u, v) for u, v in ego.edges() 
                           if u in nodes_1hop and v in nodes_1hop]
    if edges_1hop_internal:
        nx.draw_networkx_edges(ego, pos, edgelist=edges_1hop_internal,
                               alpha=0.85, width=1.8, edge_color="#ff7f0e", ax=ax)

    # Рёбра от центра
    edges_center = [(center_node, n) for n in nodes_1hop]
    nx.draw_networkx_edges(ego, pos, edgelist=edges_center,
                           alpha=0.95, width=2.5, edge_color="#d62728", ax=ax)

    # Остальные рёбра
    other_edges = [e for e in ego.edges() 
                   if e not in edges_1hop_internal and e not in edges_center]
    nx.draw_networkx_edges(ego, pos, edgelist=other_edges,
                           alpha=0.25, width=0.8, edge_color="gray", ax=ax)

    # Узлы
    if nodes_2hop:
        nx.draw_networkx_nodes(ego, pos, nodelist=list(nodes_2hop),
                               node_size=180, node_color="#c6dbef", ax=ax)
    nx.draw_networkx_nodes(ego, pos, nodelist=list(nodes_1hop),
                           node_size=320, node_color="#ffbb78", ax=ax)
    nx.draw_networkx_nodes(ego, pos, nodelist=[center_node],
                           node_size=900, node_color="#d62728", ax=ax)

    # Подписи
    labels = {center_node: str(center_node)}
    nx.draw_networkx_labels(ego, pos, labels, font_size=11, font_weight='bold', ax=ax)

    # Заголовок
    deg_center = G.degree(center_node)
    clustering_center = nx.clustering(G, center_node)
    n_1hop = len(nodes_1hop)
    ax.set_title(
        f"Ego-network of node {center_node} (degree={deg_center}, "
        f"1-hop neighbors={n_1hop}, clustering={clustering_center:.3f}) — "
        f"radius={radius} | {name.upper()}"
    )
    ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return fig


def plot_communities_on_graph(
    G: nx.Graph | nx.DiGraph,
    name: Optional[str] = None,
    target_viz_nodes: int = 1000,
    show: bool = True,
    save: bool = True,
) -> plt.Figure:
    """Communities visualization using only the largest communities for readability."""
    if name is None:
        name = "custom_graph"

    G_u = G.to_undirected() if G.is_directed() else G

    try:
        partition = community_louvain.best_partition(G_u)
    except Exception:
        communities = list(nx.community.greedy_modularity_communities(G_u))
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i

    comm_to_nodes = defaultdict(list)
    for node, cid in partition.items():
        comm_to_nodes[cid].append(node)

    sorted_comms = sorted(comm_to_nodes.items(), key=lambda x: len(x[1]), reverse=True)

    selected_nodes = []
    for _, nodes in sorted_comms:
        if len(selected_nodes) + len(nodes) > target_viz_nodes:
            break
        selected_nodes.extend(nodes)

    H = G_u.subgraph(selected_nodes).copy()

    n_communities = len(set(partition[n] for n in selected_nodes))
    print(f"→ Visualizing {len(H):,} nodes from {n_communities} largest communities")

    cmap = plt.cm.get_cmap('tab20', n_communities)
    node_colors = [cmap(partition[node]) for node in H.nodes()]

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(H, seed=42, k=0.35, iterations=60)

    nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=45, alpha=0.95, ax=ax)
    nx.draw_networkx_edges(H, pos, alpha=0.12, width=0.6, ax=ax)

    ax.set_title(f"Communities on Graph — {name.upper()} ({n_communities} largest communities)")
    ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return fig


def visualize_all_graphs(
    G: nx.Graph | nx.DiGraph,
    name: Optional[str] = None,
    show: bool = True,
    save: bool = True,
) -> None:
    """Generate all practical graph visualizations."""
    if name is None:
        name = "custom_graph"
    print(f"Generating visualizations for {name}...")

    plot_degree_rank(G, name, show=show, save=save)
    plot_ego_network(G, name, show=show, save=save)
    plot_communities_on_graph(G, name, show=show, save=save)

    print(f"✅ All practical visualizations for {name} completed.\n")