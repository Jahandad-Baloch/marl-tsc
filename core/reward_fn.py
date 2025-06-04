"""Registry access for reward functions."""
from .rewards import get_reward_fn, available_rewards, register_reward, RewardFunction

__all__ = [
    "get_reward_fn",
    "available_rewards",
    "register_reward",
    "RewardFunction",
]
