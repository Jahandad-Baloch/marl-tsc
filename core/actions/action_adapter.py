# marl_tsc/core/action_adapter.py
# This file is part of the MARL-TSC package.
# This action adapter module provides a set of classes that map high-level agent actions to low-level SUMO commands.
# It includes various action adapters for different action spaces and strategies.

from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym

class ActionAdapter(ABC):
    """Maps high-level agent actions → low-level SUMO commands."""
    @abstractmethod
    def space(self, tlc) -> gym.Space: ...
    @abstractmethod
    def apply(self, tlc, action): ...

# ------------------------------------ #
#  Concrete adapters
# ------------------------------------ #

class BinaryPhaseSwitch(ActionAdapter):
    """Select next phase index using binary action space."""
    def space(self, tlc):
        return gym.spaces.MultiBinary(len(tlc.phases))
    def apply(self, tlc, act):
        # find the first active phase
        for i in range(len(act)):
            # if the action vector is a binary vector, find the first active phase
            # (i.e., the first 1 in the vector)
            # if the action vector is a one-hot vector, find the index of the 1
            # (i.e., the index of the active phase)
            if act[i]:
                tlc.set_phase(i)
                break
        else:
            raise ValueError("No active phase found in action vector.")

class DiscretePhaseSwitch(ActionAdapter):
    """Select next phase index directly (classic CoLight style)."""
    def space(self, tlc):
        return gym.spaces.Discrete(len(tlc.phases))
    def apply(self, tlc, act):
        tlc.set_phase(int(act))

class DiscreteGreenDelta(ActionAdapter):
    """{−Δ, 0, +Δ} green-time tweak used in current code."""
    def __init__(self, delta_list=(-5., 0., 5.)):
        self.delta = np.asarray(delta_list, dtype=float)
    def space(self, tlc):
        return gym.spaces.Discrete(len(self.delta))
    def apply(self, tlc, act):
        tlc.shift_green(self.delta[int(act)])

class ContinuousGreenDelta(ActionAdapter):
    """Bounded continuous adjustment."""
    def __init__(self, low=-10., high=10.):
        self.low, self.high = low, high
    def space(self, tlc):
        return gym.spaces.Box(
            low=np.array([self.low], dtype=np.float32),
            high=np.array([self.high], dtype=np.float32)
        )
    def apply(self, tlc, act):
        tlc.shift_green(float(act[0]))

class HybridPhaseAndDelta(ActionAdapter):
    """
    Two-component hybrid:
      1) discrete next-phase id
      2) continuous green tweak
    RLlib & TorchRL handle this with nested spaces :contentReference[oaicite:4]{index=4}.
    """
    def __init__(self, delta_bounds=(-5., 5.)):
        self.low, self.high = delta_bounds
    def space(self, tlc):
        return gym.spaces.Dict({
            "phase": gym.spaces.Discrete(len(tlc.phases)),
            "delta": gym.spaces.Box(
                low=np.array([self.low], dtype=np.float32),
                high=np.array([self.high], dtype=np.float32))
        })
    def apply(self, tlc, act):
        tlc.set_phase(int(act["phase"]))
        tlc.shift_green(float(act["delta"][0]))

class MultiLevelAdapter(ActionAdapter):
    """
    Multi-level hierarchy (used by skill-hierarchy or option agents :contentReference[oaicite:5]{index=5}):
      - high-level option index
      - low-level params (optional)
    """
    def __init__(self, num_options=4):
        self.num_options = num_options
    def space(self, tlc):
        return gym.spaces.MultiDiscrete([self.num_options, len(tlc.phases)])
    def apply(self, tlc, act):
        option, phase = act
        # user-defined option dispatch
        if option == 0:  # 'normal'
            tlc.set_phase(int(phase))
        elif option == 1:  # 'fast-clear'
            tlc.fast_clear(phase)
        elif option == 2:  # 'green-wave'
            tlc.green_wave(phase)
        elif option == 3:  # 'emergency'
            tlc.emergency(phase)
        else:
            raise ValueError(f"Unknown option: {option}")
        
_registry = {
    "binary": BinaryPhaseSwitch,
    "discrete": DiscretePhaseSwitch,
    "discrete-green-delta": DiscreteGreenDelta,
    "continuous-green-delta": ContinuousGreenDelta,
    "hybrid-phase-and-delta": HybridPhaseAndDelta,
    "multi-level": MultiLevelAdapter
}

def get_action_adapter(name: str) -> ActionAdapter:
    """
    Get the action adapter class by name.
    :param name: Name of the action adapter.
    :return: ActionAdapter class.
    """
    if name not in _registry:
        raise ValueError(f"Unknown action adapter: {name}")
    return _registry[name]()

def get_action_adapter_names() -> list:
    """
    Get the names of all available action adapters.
    :return: List of action adapter names.
    """
    return list(_registry.keys())