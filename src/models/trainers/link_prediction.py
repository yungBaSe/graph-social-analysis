# src/models/trainers/link_prediction.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Any
import numpy as np

from .base import BaseTrainer
from ..utils import set_seed


class LinkPredictionTrainer(BaseTrainer):
    def __init__(self, model, lr: float = 0.01, weight_decay: float = 5e-4, seed: int = 42):
        super().__init__(model, lr, weight_decay, seed)
        self.decoder = None          # создадим позже, после определения размерности
        self.emb_dim = None

    def _ensure_decoder(self, data: Data):
        """Создаёт decoder после получения реальной размерности эмбеддингов."""
        if self.decoder is not None:
            return
        self.model.eval()
        with torch.no_grad():
            # Пробный forward на устройстве модели
            device = next(self.model.parameters()).device
            data = data.to(device)
            try:
                sample_emb = self.model(data, return_embeddings=True)
            except TypeError:
                # Если модель не поддерживает return_embeddings, возвращаем эмбеддинги другим способом
                # В GCNII/SGC просто вызовем forward и вручную отделим последний слой
                sample_emb = self._get_embeddings_fallback(data)
            self.emb_dim = sample_emb.shape[1]

        self.decoder = nn.Sequential(
            nn.Linear(self.emb_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        ).to(self.device)
        self.model.train()

    def _get_embeddings_fallback(self, data: Data) -> torch.Tensor:
        """Fallback для моделей, которые не реализуют return_embeddings."""
        # Для SGC и GCNII – можно взять выход перед классификатором.
        # Временно: просто выбросим ошибку с рекомендацией.
        raise NotImplementedError(
            f"Model {type(self.model).__name__} does not support return_embeddings. "
            "Please implement return_embeddings in the model or use a different model."
        )

    def fit(self, data: Data, epochs: int = 200, early_stopping: int = 20, verbose: bool = True) -> Dict[str, Any]:
        set_seed(self.seed)
        self.model = self.model.to(self.device)
        data = data.to(self.device)
        self._ensure_decoder(data)   # создаём decoder

        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        best_val_auc = 0.0
        patience = early_stopping
        best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.decoder.train()
            optimizer.zero_grad()

            node_emb = self.model(data, return_embeddings=True)
            pos_edge = data.train_pos
            neg_edge = data.train_neg

            pos_emb = torch.cat([node_emb[pos_edge[0]], node_emb[pos_edge[1]]], dim=1)
            neg_emb = torch.cat([node_emb[neg_edge[0]], node_emb[neg_edge[1]]], dim=1)

            pos_score = self.decoder(pos_emb).squeeze()
            neg_score = self.decoder(neg_emb).squeeze()

            scores = torch.cat([pos_score, neg_score])
            labels = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))]).to(self.device)

            loss = F.binary_cross_entropy_with_logits(scores, labels)
            loss.backward()
            optimizer.step()

            # Валидация на val
            val_metrics = self.evaluate(data, stage="val")
            val_auc = val_metrics["auc"]
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience = early_stopping
                best_state = {
                    'model': self.model.state_dict(),
                    'decoder': self.decoder.state_dict()
                }
            else:
                patience -= 1

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}")

            if patience == 0:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                # восстановить лучшие веса
                if best_state:
                    self.model.load_state_dict(best_state['model'])
                    self.decoder.load_state_dict(best_state['decoder'])
                break

        return {"status": "finished", "best_val_auc": best_val_auc}

    @torch.no_grad()
    def evaluate(self, data: Data, stage: str = "test") -> Dict[str, float]:
        self.model.eval()
        self.decoder.eval()
        data = data.to(self.device)
        self._ensure_decoder(data)

        node_emb = self.model(data, return_embeddings=True)

        if stage == "val":
            pos_edge = data.val_pos
            neg_edge = data.val_neg
        else:
            pos_edge = data.test_pos
            neg_edge = data.test_neg

        pos_emb = torch.cat([node_emb[pos_edge[0]], node_emb[pos_edge[1]]], dim=1)
        neg_emb = torch.cat([node_emb[neg_edge[0]], node_emb[neg_edge[1]]], dim=1)

        pos_score = self.decoder(pos_emb).squeeze()
        neg_score = self.decoder(neg_emb).squeeze()

        scores = torch.cat([pos_score, neg_score]).cpu().numpy()
        y_true = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))]).numpy()

        auc = roc_auc_score(y_true, scores)
        ap = average_precision_score(y_true, scores)

        # Hits@K и MRR
        all_scores = np.concatenate([pos_score.cpu().numpy(), neg_score.cpu().numpy()])
        all_labels = np.concatenate([np.ones_like(pos_score.cpu().numpy()), np.zeros_like(neg_score.cpu().numpy())])
        sorted_indices = np.argsort(-all_scores)
        sorted_labels = all_labels[sorted_indices]
        pos_indices = np.where(sorted_labels == 1)[0]
        hits_at_k = {}
        for k in [20, 50, 100]:
            hits_at_k[f"hits_{k}"] = (pos_indices < k).mean() if len(pos_indices) > 0 else 0.0
        mrr = (1.0 / (pos_indices + 1)).mean() if len(pos_indices) > 0 else 0.0

        return {
            "auc": float(auc),
            "ap": float(ap),
            **hits_at_k,
            "mrr": float(mrr)
        }