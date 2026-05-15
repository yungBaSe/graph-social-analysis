# src/models/base.py
from typing import Dict, Any, Optional
import networkx as nx
import torch
import pickle
import os

class BaseModel:
    """Базовый класс для всех моделей."""

    def __init__(self, name: str = "base_model", seed: int = 42, undirected: bool = True, **kwargs):
        self.name = name
        self.seed = seed
        self.undirected = undirected
        self.is_fitted = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, **kwargs) -> None:
        """Обучить модель. GNN‑модели тренируются через Trainer, этот метод не используется."""
        raise NotImplementedError(f"{self.__class__.__name__} не поддерживает прямой вызов fit(). Используйте Trainer.")

    def predict(self, **kwargs) -> torch.Tensor:
        """Предсказать. GNN‑модели предсказывают через Trainer.evaluate()."""
        raise NotImplementedError(f"{self.__class__.__name__} не поддерживает прямой вызов predict(). Используйте Trainer.")

    def forward(self, data) -> torch.Tensor:
        """Для PyTorch моделей (GNN)."""
        raise NotImplementedError("Метод forward должен быть переопределён в PyTorch‑моделях")

    def save(self, path: str) -> None:
        """Сохранение модели."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if isinstance(self, torch.nn.Module):
            # Для GNN моделей
            torch.save({
                'model_state_dict': self.state_dict(),
                'name': self.name,
                'seed': self.seed,
                'undirected': self.undirected,
            }, path)
        else:
            # Для sklearn моделей
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        
        print(f"✅ Model saved → {path}")

    def load(self, path: str) -> None:
        """Загрузка модели."""
        if isinstance(self, torch.nn.Module):
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            with open(path, 'rb') as f:
                loaded = pickle.load(f)
                self.__dict__.update(loaded.__dict__)
        
        self.is_fitted = True
        print(f"✅ Model loaded ← {path}")