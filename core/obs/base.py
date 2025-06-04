import abc
import gymnasium as gym
from typing import Dict

class BaseObservationBuilder(abc.ABC):
    """
    Collects **raw** features from SUMO or pre-computed tensors.
    Should perform no transformation except basic aggregation.
    """

    def __init__(self, tls_id, cfg, traci_if, logger=None):
        self.id = tls_id
        self.cfg = cfg
        self.traci = traci_if
        self.log = logger

    @abc.abstractmethod
    def space(self) -> gym.Space:
        """Return a Gymnasium space object describing the raw vector."""
        ...

    @abc.abstractmethod
    def __call__(self) -> Dict[str, float]:
        """Return a dict (or np.ndarray) of raw features for ONE step."""
        ...
