# marl_tsc/core/obs_builder.py
from abc import ABC, abstractmethod
import numpy as np, gymnasium as gym

class ObservationBuilder(ABC):
    """Collects raw features for a single traffic-light controller."""
    def __init__(self, tlc, cfg):
        self.tlc, self.cfg = tlc, cfg

    @abstractmethod
    def space(self) -> gym.Space: ...

    @abstractmethod
    def build(self) -> np.ndarray: ...

# ---------- Concrete builders ---------- #

class QueueLenWaitTime(ObservationBuilder):
    """queue, waiting-time, phase index (scalar or one-hot)"""
    def __init__(self, tlc, cfg):
        super().__init__(tlc, cfg)
        self.use_onehot = cfg.metrics.get("use_phase_one_hot", False)

    def space(self):
        dim = 3 if not self.use_onehot else 2 + len(self.tlc.phases)
        return gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)

    def build(self):
        q = self.tlc.data_collector.get_queue_length()
        w = self.tlc.data_collector.get_wait_time()
        phase = self.tlc.get_current_phase_one_hot() if self.use_onehot else [self.tlc.current_phase]
        return np.asarray([q, w, *phase], dtype=np.float32)

class FullDetectorSet(ObservationBuilder):
    """All detector metrics supported in SUMO e1/e2 (throughput, speed, occ, jam)."""
    def space(self):
        return gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
    def build(self):
        data = self.tlc.data_collector.fetch_traffic_data()  # existing helper
        return np.asarray([data[k] for k in
            ("throughput","mean_speed","occupancy","queue_length","halt_count","waiting_time")],
            dtype=np.float32)

# ---------- Registry ---------- #
_registry = {
    "queue_basic": QueueLenWaitTime,
    "detectors_full": FullDetectorSet
}

def get_builder(name, tlc, cfg):
    if name not in _registry:
        raise KeyError(f"Unknown ObservationBuilder {name}")
    return _registry[name](tlc, cfg)

def get_builder_names():
    """Get all available observation builder names."""
    return list(_registry.keys())