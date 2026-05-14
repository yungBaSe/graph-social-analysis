# src/embeddings/graph_embeddings.py
import pickle
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Any
from tqdm import tqdm

import networkx as nx
import numpy as np

from src.data.data_loader import get_dataset


# === PATH CONFIGURATION ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def _get_cache_path(name: str, method: str, dimensions: int, params: dict) -> Path:
    """Create stable cache key based on all important parameters."""
    param_str = json.dumps(params, sort_keys=True)
    key = hashlib.md5(f"{method}_{dimensions}_{param_str}".encode()).hexdigest()[:12]
    return DATA_PROCESSED / f"{name}_{method}_d{dimensions}_{key}.pkl"


def node2vec_embeddings(
    G: nx.Graph | nx.DiGraph,
    dimensions: int = 128,
    walk_length: int = 80,
    num_walks: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    window: int = 10,
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """Node2Vec embeddings with explicit parameters."""
    from node2vec import Node2Vec

    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=1,
        seed=seed,
    )
    model = node2vec.fit(window=window, min_count=1, batch_words=4)
    embeddings = {int(node): model.wv[str(node)] for node in G.nodes()}
    return embeddings


def random_walk_embeddings(
    G: nx.Graph | nx.DiGraph,
    dimensions: int = 128,
    walk_length: int = 80,
    num_walks: int = 10,
    window: int = 10,
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """Simple DeepWalk-style random walk embeddings."""
    import random
    from gensim.models import Word2Vec

    random.seed(seed)
    nodes = list(G.nodes())
    walks = []

    for _ in tqdm(range(num_walks), desc="Generating walks", leave=False):
        for node in nodes:
            walk = [node]
            current = node
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break
                current = random.choice(neighbors)
                walk.append(current)
            walks.append([str(n) for n in walk])

    model = Word2Vec(
        walks,
        vector_size=dimensions,
        window=window,
        min_count=1,
        sg=1,
        workers=1,
        seed=seed,
    )
    embeddings = {int(node): model.wv[str(node)] for node in G.nodes() if str(node) in model.wv}
    return embeddings

def grace_embeddings(
    G: nx.Graph | nx.DiGraph,
    dimensions: int = 128,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """
    Self-supervised эмбеддинги методом GRACE.
    """
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    from torch_geometric.utils import to_undirected

    # Подготовка данных
    G_u = G.to_undirected() if isinstance(G, nx.DiGraph) else G
    nodes = list(G_u.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edge_index = torch.tensor(
        [[node_to_idx[u], node_to_idx[v]] for u, v in G_u.edges()],
        dtype=torch.long
    ).t().contiguous() if G_u.number_of_edges() > 0 else torch.zeros((2, 0), dtype=torch.long)

    # Если есть фичи, используем их; иначе — единичную матрицу
    if hasattr(G_u, 'features') and G_u.features is not None:
        x = torch.tensor(G_u.features, dtype=torch.float32)
    else:
        x = torch.eye(len(nodes), dtype=torch.float32)

    data = Data(x=x, edge_index=to_undirected(edge_index))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    class GRACE(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.encoder = torch.nn.ModuleList([
                GCNConv(in_dim, hidden_dim),
                GCNConv(hidden_dim, out_dim)
            ])
        def forward(self, x, edge_index):
            h = F.relu(self.encoder[0](x, edge_index))
            return self.encoder[1](h, edge_index)

    model = GRACE(x.size(1), dimensions, dimensions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Функция InfoNCE
    def info_nce(z1, z2, tau=0.2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim = torch.mm(z1, z2.t()) / tau
        labels = torch.arange(sim.size(0)).to(sim.device)
        loss = F.cross_entropy(sim, labels)
        return loss

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # Два варианта с дропаутом
        z1 = model(data.x, data.edge_index)
        z2 = model(data.x, data.edge_index)
        loss = info_nce(z1, z2)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index).cpu().numpy()
    embeddings = {node: z[i] for i, node in enumerate(nodes)}
    return embeddings

def dgi_embeddings(
    G: nx.Graph | nx.DiGraph,
    dimensions: int = 128,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """
    Self-supervised эмбеддинги методом Deep Graph Infomax (DGI).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    from torch_geometric.nn import GCNConv, DeepGraphInfomax
    from torch_geometric.data import Data
    from torch_geometric.utils import to_undirected

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    G_u = G.to_undirected() if isinstance(G, nx.DiGraph) else G
    nodes = list(G_u.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edge_index = torch.tensor(
        [[node_to_idx[u], node_to_idx[v]] for u, v in G_u.edges()],
        dtype=torch.long
    ).t().contiguous() if G_u.number_of_edges() > 0 else torch.zeros((2, 0), dtype=torch.long)

    x = torch.tensor(G_u.features if hasattr(G_u, 'features') else np.eye(len(nodes)), dtype=torch.float32)
    data = Data(x=x, edge_index=to_undirected(edge_index))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    class Encoder(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)
        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            return self.conv2(x, edge_index)

    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    encoder = Encoder(data.x.size(1), dimensions, dimensions).to(device)
    model = DeepGraphInfomax(
        hidden_channels=dimensions,
        encoder=encoder,
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(data.x, data.edge_index)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z = encoder(data.x, data.edge_index).cpu().detach().numpy()
    embeddings = {node: z[i] for i, node in enumerate(nodes)}
    return embeddings


def compute_embeddings(
    G: nx.Graph | nx.DiGraph,
    method: str = "node2vec",
    dimensions: int = 128,
    name: Optional[str] = None,
    seed: int = 42,
    **kwargs,
) -> Dict[int, np.ndarray]:
    """Main function. Compute embeddings for any graph."""
    if name:
        cache_path = _get_cache_path(name, method, dimensions, kwargs)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

    if method == "node2vec":
        embeddings = node2vec_embeddings(
            G, dimensions=dimensions, seed=seed, **kwargs
        )
    elif method == "random_walk":
        embeddings = random_walk_embeddings(
            G, dimensions=dimensions, seed=seed, **kwargs
        )
    elif method == "grace":
        embeddings = grace_embeddings(G, dimensions=dimensions, seed=seed, **kwargs)
    elif method == "dgi":
        embeddings = dgi_embeddings(G, dimensions=dimensions, seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown embedding method: {method}. Use 'node2vec' or 'random_walk'.")

    if name:
        cache_path = _get_cache_path(name, method, dimensions, kwargs)
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"✅ Embeddings cached: {name} | {method} | dim={dimensions}")

    return embeddings


def compute_dataset_embeddings(
    name: str,
    method: str = "node2vec",
    dimensions: int = 128,
    seed: int = 42,
    **kwargs,
) -> Dict[int, np.ndarray]:
    """Convenience wrapper for dataset by name."""
    dataset = get_dataset(name, verbose=False)
    G = dataset["graph"]
    return compute_embeddings(
        G=G,
        method=method,
        dimensions=dimensions,
        name=name,
        seed=seed,
        **kwargs,
    )


def load_embeddings(
    name: str,
    method: str = "node2vec",
    dimensions: int = 128,
    seed: int = 42,
    **kwargs,
) -> Dict[int, np.ndarray]:
    """Load cached embeddings."""
    cache_path = _get_cache_path(name, method, dimensions, kwargs)
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    raise FileNotFoundError(f"No cached embeddings for {name}")


def list_cached_embeddings() -> None:
    """List all cached embedding files."""
    files = sorted(DATA_PROCESSED.glob("*_d*_*.pkl"))
    print(f"Found {len(files)} cached embedding files:")
    for f in files:
        print(f"  • {f.name}")