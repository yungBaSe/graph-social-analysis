# src/models/registry.py
from typing import Dict, Type, Any
from .base import BaseModel

_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(name: str):
    """Декоратор для регистрации моделей."""
    def decorator(cls: Type[BaseModel]):
        _MODEL_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_model(model_name: str, **kwargs) -> BaseModel:
    """Создаёт экземпляр модели по имени."""
    model_name = model_name.lower()
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(_MODEL_REGISTRY.keys())}")
    
    model_class = _MODEL_REGISTRY[model_name]
    return model_class(**kwargs)


def list_models() -> list:
    """Возвращает список всех зарегистрированных моделей."""
    return list(_MODEL_REGISTRY.keys())