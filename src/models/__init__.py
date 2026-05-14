# src/models/__init__.py
from .registry import get_model, register_model, list_models
from .data import make_node_classification_data, make_link_prediction_data, prepare_graph
from .trainers import BaseTrainer, NodeClassificationTrainer, LinkPredictionTrainer
from .baselines import LogisticBaseline, MLPBaseline
from .gnn import (
    GCN, GraphSAGE, GAT, GATv2,
    GraphTransformer, GIN, SGC, JKNet, GCNII
)
from .community import CommunityDetector, detect_communities

__all__ = [
    "get_model", "register_model", "list_models",
    "make_node_classification_data", "make_link_prediction_data", "prepare_graph",
    "BaseTrainer", "NodeClassificationTrainer", "LinkPredictionTrainer",
    "LogisticBaseline", "MLPBaseline",
    "GCN", "GraphSAGE", "GAT", "GATv2",
    "GraphTransformer", "GIN", "SGC", "JKNet", "GCNII",
    "CommunityDetector", "detect_communities",
]