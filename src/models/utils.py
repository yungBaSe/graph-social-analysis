# src/models/utils.py
import torch
import numpy as np
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """Фиксируем все случайности."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Возвращает CUDA или CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_torch_tensor(x: Optional[np.ndarray] = None) -> Optional[torch.Tensor]:
    """Конвертирует numpy массив в torch.Tensor."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, dtype=torch.float32)