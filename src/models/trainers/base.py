# src/models/trainers/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch


class BaseTrainer(ABC):
    """Базовый класс для всех Trainer'ов."""

    def __init__(self, model, lr: float = 0.01, weight_decay: float = 5e-4, seed: int = 42):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = None

    @abstractmethod
    def fit(self, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def evaluate(self, **kwargs) -> Dict[str, float]:
        pass