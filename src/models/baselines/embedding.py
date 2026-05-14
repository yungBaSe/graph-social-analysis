# src/models/baselines/embedding.py
import torch
import numpy as np
from typing import Dict, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from ..base import BaseModel
from ..registry import register_model
from ..utils import set_seed


@register_model("logistic")
class LogisticBaseline(BaseModel):
    """Logistic Regression на эмбеддингах / features."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, **kwargs):
        super().__init__(name="logistic", **kwargs)
        self.C = C
        self.max_iter = max_iter
        self.model = None

    def fit(self, X=None, y=None, embeddings=None, node_features=None, labels=None, **kwargs):
        set_seed(self.seed)

        if X is not None:
            # Вызов из Trainer
            self.model = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.seed, n_jobs=-1)
            self.model.fit(X, y)
        elif embeddings is not None:
            X = np.array([embeddings[n] for n in sorted(embeddings.keys())])
            self.model = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.seed, n_jobs=-1)
            self.model.fit(X, labels)
        elif node_features is not None:
            self.model = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.seed, n_jobs=-1)
            self.model.fit(node_features, labels)
        else:
            raise ValueError("Provide X/y or embeddings/labels or node_features/labels")

        self.is_fitted = True

    def predict(self, X=None, embeddings=None, node_features=None):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet")

        if X is not None:
            return self.model.predict(X)
        elif embeddings is not None:
            X = np.array([embeddings[n] for n in sorted(embeddings.keys())])
            return self.model.predict(X)
        elif node_features is not None:
            return self.model.predict(node_features)
        else:
            raise ValueError("Provide data for prediction")

    def forward(self, *args):
        raise NotImplementedError("LogisticBaseline is not a PyTorch model")


@register_model("mlp")
class MLPBaseline(BaseModel):
    """Multi-Layer Perceptron на эмбеддингах / features."""

    def __init__(self, hidden_layer_sizes: tuple = (128, 64), max_iter: int = 500, **kwargs):
        super().__init__(name="mlp", **kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.model = None

    def fit(self, X=None, y=None, embeddings=None, node_features=None, labels=None, **kwargs):
        set_seed(self.seed)

        if X is not None:
            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                max_iter=self.max_iter,
                random_state=self.seed,
                early_stopping=True,
                validation_fraction=0.1,
            )
            self.model.fit(X, y)
        elif embeddings is not None:
            X = np.array([embeddings[n] for n in sorted(embeddings.keys())])
            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                max_iter=self.max_iter,
                random_state=self.seed,
                early_stopping=True,
                validation_fraction=0.1,
            )
            self.model.fit(X, labels)
        elif node_features is not None:
            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                max_iter=self.max_iter,
                random_state=self.seed,
                early_stopping=True,
                validation_fraction=0.1,
            )
            self.model.fit(node_features, labels)
        else:
            raise ValueError("Provide X/y or embeddings/labels or node_features/labels")

        self.is_fitted = True

    def predict(self, X=None, embeddings=None, node_features=None):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet")

        if X is not None:
            return self.model.predict(X)
        elif embeddings is not None:
            X = np.array([embeddings[n] for n in sorted(embeddings.keys())])
            return self.model.predict(X)
        elif node_features is not None:
            return self.model.predict(node_features)
        else:
            raise ValueError("Provide data for prediction")

    def forward(self, *args):
        raise NotImplementedError("MLPBaseline is not a PyTorch model")