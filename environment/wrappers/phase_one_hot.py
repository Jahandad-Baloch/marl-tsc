

    # def get_current_phase_one_hot(self):
    #     """
    #     Return the current phase as a one-hot encoded vector.
    #     """
    #     self.current_phase = self.sumo_interface.trafficlight.getPhase(self.id)
    #     phase_index = self.current_phase
    #     num_phases = len(self.phases)
    #     one_hot = np.zeros(num_phases, dtype=float)  # Use float dtype for PyTorch compatibility
    #     one_hot[phase_index] = 1.0
    #     return one_hot

import gymnasium as gym
from ..registry import register_wrapper

@register_wrapper("phase_one_hot")
class PhaseOneHotWrapper(gym.ObservationWrapper):
    """
    Appends one-hot of current TLS phase to the low-level builder vector.
    Requires that env exposes `env.get_current_phase(tls_id)`.
    """
    def __init__(self, env):
        super().__init__(env)
        self.tls_id = env.tls_id
        self.use_onehot = env.cfg.metrics.get("use_phase_one_hot", False)
        if not self.use_onehot:
            raise ValueError("Phase one-hot encoding is not enabled in the configuration.")
        self.observation_space = gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(self.observation_space.shape[0] + len(env.phases),),
            dtype=self.observation_space.dtype
        )