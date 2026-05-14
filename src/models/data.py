# src/models/data.py
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from typing import Optional, Dict, Any
import scipy.sparse as sp


def prepare_graph(G: nx.Graph | nx.DiGraph, undirected: bool = True) -> nx.Graph:
    if undirected and isinstance(G, nx.DiGraph):
        return G.to_undirected()
    return G


def make_node_classification_data(
    G: nx.Graph | nx.DiGraph,
    labels: Optional[np.ndarray] = None,
    node_features: Optional[np.ndarray] = None,  # теперь только плотный numpy
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    undirected: bool = True,
    masks: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Готовит данные для node classification. Если переданы masks, использует их."""
    if labels is None:
        raise ValueError("Для Node Classification необходимы labels.")

    G = prepare_graph(G, undirected=undirected)
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    y_np = np.array(labels)
    unique_labels = np.unique(y_np)
    label_to_idx = {old: new for new, old in enumerate(unique_labels)}
    y_mapped = np.array([label_to_idx[label] for label in y_np])
    y = torch.tensor(y_mapped, dtype=torch.long)
    n_classes = len(unique_labels)

    # Node features
    if node_features is not None:
        x = torch.tensor(node_features, dtype=torch.float32)
    else:
        x = torch.ones((n_nodes, 1), dtype=torch.float32)

    # Edge index
    edge_index = torch.tensor(
        [[node_to_idx[u], node_to_idx[v]] for u, v in G.edges()],
        dtype=torch.long
    ).t().contiguous()

    # Маски: если предоставлены готовые, используем их
    if masks is not None:
        train_mask = torch.tensor(masks['train_mask'], dtype=torch.bool)
        val_mask = torch.tensor(masks['val_mask'], dtype=torch.bool)
        test_mask = torch.tensor(masks['test_mask'], dtype=torch.bool)
    else:
        indices = np.arange(n_nodes)
        class_counts = np.bincount(y_mapped)
        can_stratify = (class_counts >= 2).all()
        if can_stratify:
            train_idx, temp_idx = train_test_split(
                indices, test_size=test_size + val_size, random_state=random_state,
                stratify=y_mapped, shuffle=True
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=test_size / (test_size + val_size),
                random_state=random_state, shuffle=True
            )
        else:
            train_idx, temp_idx = train_test_split(
                indices, test_size=test_size + val_size, random_state=random_state, shuffle=True
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=test_size / (test_size + val_size),
                random_state=random_state, shuffle=True
            )

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return {
        "data": data,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "node_to_idx": node_to_idx,
        "n_classes": n_classes,
    }


def make_link_prediction_data(
    G: nx.Graph | nx.DiGraph,
    embeddings: Optional[Dict[int, np.ndarray]] = None,
    node_features: Optional[np.ndarray] = None,
    test_ratio: float = 0.1,
    val_ratio: float = 0.05,
    random_state: int = 42,
    undirected: bool = True,
    neg_multiplier: int = 1,
) -> Dict[str, Any]:
    G = prepare_graph(G, undirected=undirected)

    edges = list(G.edges())
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    train_edges, temp_edges = train_test_split(
        edges, test_size=(test_ratio + val_ratio), random_state=random_state, shuffle=True
    )
    val_edges, test_edges = train_test_split(
        temp_edges, test_size=test_ratio / (test_ratio + val_ratio),
        random_state=random_state, shuffle=True
    )

    if node_features is not None:
        x = torch.tensor(node_features, dtype=torch.float32)
    elif embeddings is not None:
        emb_list = [embeddings[node] for node in nodes]
        x = torch.tensor(np.array(emb_list), dtype=torch.float32)
    else:
        x = torch.ones((n_nodes, 1), dtype=torch.float32)

    def edges_to_tensor(edge_list):
        return torch.tensor(
            [[node_to_idx[u], node_to_idx[v]] for u, v in edge_list],
            dtype=torch.long
        ).t().contiguous()

    train_pos = edges_to_tensor(train_edges)
    val_pos = edges_to_tensor(val_edges)
    test_pos = edges_to_tensor(test_edges)

    def sample_negative_edges(num_pos: int) -> torch.Tensor:
        num_neg = int(num_pos * neg_multiplier)
        neg = torch.randint(0, n_nodes, (2, num_neg), dtype=torch.long)
        return neg

    train_neg = sample_negative_edges(train_pos.size(1))
    val_neg = sample_negative_edges(val_pos.size(1))
    test_neg = sample_negative_edges(test_pos.size(1))

    edge_index = torch.cat([train_pos, train_neg], dim=1)
    edge_label = torch.cat([
        torch.ones(train_pos.size(1), dtype=torch.long),
        torch.zeros(train_neg.size(1), dtype=torch.long)
    ])

    data = Data(x=x, edge_index=edge_index)
    data.edge_label = edge_label
    data.train_pos = train_pos
    data.train_neg = train_neg
    data.val_pos = val_pos
    data.val_neg = val_neg
    data.test_pos = test_pos
    data.test_neg = test_neg

    return {
        "data": data,
        "train_pos": train_pos,
        "train_neg": train_neg,
        "val_pos": val_pos,
        "val_neg": val_neg,
        "test_pos": test_pos,
        "test_neg": test_neg,
        "node_to_idx": node_to_idx,
    }