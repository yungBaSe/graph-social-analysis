# src/embeddings/__init__.py
from .graph_embeddings import (
    compute_embeddings,
    node2vec_embeddings,
    random_walk_embeddings,
    grace_embeddings,
    dgi_embeddings,
)

__all__ = [
    "compute_embeddings",
    "node2vec_embeddings",
    "random_walk_embeddings",
    "grace_embeddings",
    "dgi_embeddings",
]