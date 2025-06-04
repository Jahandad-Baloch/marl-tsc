"""Simple registry utilities for observation builders."""
from typing import Dict, Type

_builder_registry: Dict[str, Type] = {}


def register_builder(name: str):
    """Decorator to register an observation builder class."""
    def deco(cls):
        _builder_registry[name] = cls
        return cls
    return deco


def get_builder_cls(name: str):
    if name not in _builder_registry:
        raise KeyError(f"Unknown builder {name}")
    return _builder_registry[name]


def available_builders():
    return list(_builder_registry.keys())
