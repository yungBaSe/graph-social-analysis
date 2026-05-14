# src/models/trainers/__init__.py
from .base import BaseTrainer
from .node_classification import NodeClassificationTrainer
from .link_prediction import LinkPredictionTrainer

__all__ = [
    "BaseTrainer",
    "NodeClassificationTrainer",
    "LinkPredictionTrainer",
]