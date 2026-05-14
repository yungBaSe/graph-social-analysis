import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import Optional

from src.embeddings.graph_embeddings import compute_dataset_embeddings
from src.data.data_loader import get_dataset
from src.visualization.utils import show_and_save, set_style


def plot_embeddings_2d(
    G: nx.Graph | nx.DiGraph,
    name: Optional[str] = None,
    method: str = "node2vec",
    dimensions: int = 128,
    seed: int = 42,
    color_by: str = "degree",
    reducer: str = "tsne",
    show: bool = True,
    save: bool = True,
    **kwargs
) -> plt.Figure:
    """2D projection of node embeddings (t-SNE / UMAP)."""
    if name is None:
        name = "custom_graph"

    embeddings_dict = compute_dataset_embeddings(
        name=name if name != "custom_graph" else None,
        method=method,
        dimensions=dimensions,
        seed=seed,
        **kwargs
    )

    node_ids = list(embeddings_dict.keys())
    X = np.array([embeddings_dict[n] for n in node_ids])

    if reducer == "umap":
        try:
            import umap
            X_2d = umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
            reducer_name = "UMAP"
        except ImportError:
            reducer = "tsne"
            reducer_name = "UMAP (fallback to t-SNE)"
    else:
        from sklearn.manifold import TSNE
        X_2d = TSNE(n_components=2, random_state=seed, perplexity=min(30, len(X)-1)).fit_transform(X)
        reducer_name = "t-SNE"

    if color_by == "labels" and name != "custom_graph":
        dataset = get_dataset(name, verbose=False)
        if dataset.get("labels") is not None:
            labels = dataset["labels"].iloc[:, 0]
            colors = [labels.get(n, -1) for n in node_ids]
            cmap = "tab10"
            cbar_label = "Node Label"
        else:
            color_by = "degree"
    if color_by == "degree":
        colors = [G.degree(n) for n in node_ids]
        cmap = "viridis"
        cbar_label = "Node Degree"

    fig, ax = plt.subplots(figsize=(11, 8))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap=cmap, s=15, alpha=0.75)
    ax.set_title(f"{method.upper()} Embeddings — {name.upper()} ({reducer_name})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.colorbar(scatter, ax=ax, label=cbar_label)

    show_and_save(fig, name, f"embeddings_{method}_{reducer}_{color_by}", show=show, save=save, dimensions=dimensions)
    return fig


def plot_embeddings_comparison(
    G: nx.Graph | nx.DiGraph,
    name: Optional[str] = None,
    methods: Optional[list] = None,
    dimensions: int = 128,
    seed: int = 42,
    color_by: str = "degree",
    reducer: str = "tsne",
    show: bool = True,
    save: bool = True,
) -> plt.Figure:
    """Side-by-side comparison of different embedding methods."""
    if name is None:
        name = "custom_graph"
    if methods is None:
        methods = ["node2vec", "random_walk"]

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6.5), squeeze=False)

    for i, method in enumerate(methods):
        ax = axes[0, i]
        
        embeddings_dict = compute_dataset_embeddings(
            name=name if name != "custom_graph" else None,
            method=method,
            dimensions=dimensions,
            seed=seed,
        )

        node_ids = list(embeddings_dict.keys())
        X = np.array([embeddings_dict[n] for n in node_ids])

        if reducer == "umap":
            try:
                import umap
                X_2d = umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
                reducer_name = "UMAP"
            except ImportError:
                from sklearn.manifold import TSNE
                X_2d = TSNE(n_components=2, random_state=seed, perplexity=min(30, len(X)-1)).fit_transform(X)
                reducer_name = "t-SNE"
        else:
            from sklearn.manifold import TSNE
            X_2d = TSNE(n_components=2, random_state=seed, perplexity=min(30, len(X)-1)).fit_transform(X)
            reducer_name = "t-SNE"

        colors = [G.degree(n) for n in node_ids]
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap="viridis", s=15, alpha=0.75)
        ax.set_title(f"{method.upper()} ({reducer_name})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.colorbar(scatter, ax=ax, label="Node Degree")

    fig.suptitle(f"Embedding Methods Comparison — {name.upper()}", fontsize=16)
    
    show_and_save(fig, name, f"embeddings_comparison_{'_'.join(methods)}", show=show, save=save, dimensions=dimensions)
    return fig