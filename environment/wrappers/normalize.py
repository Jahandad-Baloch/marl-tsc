# envs/wrappers/normalize_obs.py
import gymnasium as gym, numpy as np, torch
from marl_tsc.core.running_stats import RunningMeanStd

class NormalizeObs(gym.ObservationWrapper):
    def __init__(self, env, clip=5.0):
        super().__init__(env)
        shape = env.single_observation_space.shape
        self.rms = RunningMeanStd(shape)
        self.clip = clip

    def observation(self, obs):
        # obs is Dict[agent_id -> np.array]
        cat = np.vstack(list(obs.values()))
        self.rms.update(cat)
        normed = {k: np.clip((v - self.rms.mean.numpy()) /
                             np.sqrt(self.rms.var.numpy() + 1e-8),
                             -self.clip, self.clip) for k, v in obs.items()}
        return normed
