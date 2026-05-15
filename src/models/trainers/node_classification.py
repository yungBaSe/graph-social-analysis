# src/models/trainers/node_classification.py
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any
import numpy as np

from src.models.trainers.base import BaseTrainer
from src.models.utils import set_seed


class NodeClassificationTrainer(BaseTrainer):
    """Trainer для Node Classification с опциональной балансировкой классов."""

    def fit(
        self,
        data: Data,
        epochs: int = 200,
        early_stopping: int = 20,
        verbose: bool = False,
        balance: bool = False,
    ) -> Dict[str, Any]:
        set_seed(self.seed)

        # === sklearn модели ===
        if not isinstance(self.model, torch.nn.Module):
            X = data.x.cpu().numpy()
            y = data.y.cpu().numpy()
            train_mask = data.train_mask.cpu().numpy()
            if balance:
                # Для sklearn передаём sample_weight
                weights = self._compute_weights(y[train_mask])
                self.model.fit(X[train_mask], y[train_mask], sample_weight=weights)
            else:
                self.model.fit(X[train_mask], y[train_mask])
            self.model.is_fitted = True
            if verbose:
                print("✅ sklearn модель обучена")
            return {"status": "fitted"}

        # === PyTorch GNN ===
        self.model = self.model.to(self.device)
        data = data.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Подготовка весов для сбалансированного CrossEntropy
        if balance:
            y_train = data.y[data.train_mask].cpu().numpy()
            class_counts = np.bincount(y_train)
            weights = 1.0 / class_counts[class_counts > 0]
            weights = torch.tensor(weights / weights.sum() * len(weights), dtype=torch.float32).to(self.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        best_val_acc = 0.0
        patience = early_stopping
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()

            val_metrics = self.evaluate(data, mask=data.val_mask)
            val_acc = val_metrics["accuracy"]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience = early_stopping
            else:
                patience -= 1
            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")
            if patience == 0:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best val acc: {best_val_acc:.4f})")
                break
        return {"best_val_acc": best_val_acc, "best_epoch": best_epoch}

    @torch.no_grad()
    def evaluate(self, data: Data, mask=None) -> Dict[str, float]:
        if not isinstance(self.model, torch.nn.Module):
            X = data.x.cpu().numpy()
            y_true = data.y[mask].cpu().numpy() if mask is not None else data.y.cpu().numpy()
            y_pred = self.model.predict(X[mask] if mask is not None else X)
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
                "f1_micro": float(f1_score(y_true, y_pred, average="micro")),
                "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
            }

        self.model.eval()
        data = data.to(self.device)
        out = self.model(data)
        pred = out.argmax(dim=-1)
        if mask is None:
            mask = data.test_mask
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "f1_micro": float(f1_score(y_true, y_pred, average="micro")),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        }

    def _compute_weights(self, y: np.ndarray) -> np.ndarray:
        """Вычисляет веса для сбалансированного обучения в sklearn."""
        class_counts = np.bincount(y)
        weights = 1.0 / class_counts[class_counts > 0]
        weights = weights / weights.sum() * len(weights)
        sample_weights = np.array([weights[i] for i in y])
        return sample_weights