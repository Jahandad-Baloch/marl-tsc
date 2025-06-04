# marl_tsc/core/registeries.py
from .actions.action_adapter import *

_action_registry = {
    "discrete_phase": DiscretePhaseSwitch,
    "green_delta": DiscreteGreenDelta,
    "continuous_delta": ContinuousGreenDelta,
    "hybrid": HybridPhaseAndDelta,
    "multilevel": MultiLevelAdapter,
}

def get_action_adapter(name: str, **kwargs) -> ActionAdapter:
    if name not in _action_registry:
        raise KeyError(f"Unknown action adapter {name}")
    return _action_registry[name](**kwargs)

def get_action_adapter_names() -> list:
    """Get all available action adapter names."""
    return list(_action_registry.keys())
