from __future__ import annotations
import abc
from typing import Dict, Any

class RewardFunction(abc.ABC):
    """Base interface for computing rewards for a single agent."""
    def __init__(self, **kwargs: Any):
        pass

    @abc.abstractmethod
    def __call__(self, prev_obs: Dict[str, float], curr_obs: Dict[str, float]) -> float:
        ...


class QueueDiffReward(RewardFunction):
    """Reward negative change in queue length."""
    metric = "queue_length"
    def __call__(self, prev_obs: Dict[str, float], curr_obs: Dict[str, float]) -> float:
        before = prev_obs.get(self.metric, 0.0)
        after = curr_obs.get(self.metric, 0.0)
        return -(after - before)


class WaitTimeDiffReward(RewardFunction):
    """Reward reduction in waiting time."""
    metric = "waiting_time"
    def __call__(self, prev_obs: Dict[str, float], curr_obs: Dict[str, float]) -> float:
        before = prev_obs.get(self.metric, 0.0)
        after = curr_obs.get(self.metric, 0.0)
        return -(after - before)


_registry: Dict[str, type[RewardFunction]] = {
    "queue_diff": QueueDiffReward,
    "wait_time_diff": WaitTimeDiffReward,
}


def register_reward(name: str):
    def deco(cls: type[RewardFunction]):
        _registry[name] = cls
        return cls
    return deco


def get_reward_fn(name: str, **kwargs: Any) -> RewardFunction:
    if name not in _registry:
        raise KeyError(f"Unknown reward function {name}")
    return _registry[name](**kwargs)


def available_rewards():
    return list(_registry.keys())
