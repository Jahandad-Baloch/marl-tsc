import numpy as np
import gymnasium as gym
from .base import BaseObservationBuilder
from ..registry import register_builder


@register_builder("lanearea")
class LaneAreaBuilder(BaseObservationBuilder):
    """Collects metrics from e2 lane area detectors."""

    def __init__(self, tls_id, cfg, traci_if, logger=None):
        super().__init__(tls_id, cfg, traci_if, logger)
        self._metrics = cfg["metrics"]["obs_metrics"]
        self._init_detectors()

    def _init_detectors(self):
        lanes = {lnk[0] for links in self.traci.trafficlight.getControlledLinks(self.id)
                          for lnk in links}
        pref = self.cfg["detectors"]["detector_prefix"]
        enabled = self.cfg["detectors"]["enabled"]
        self.detectors = [f"{pref[tp]}_{ln}" for tp in enabled for ln in lanes if tp.startswith("e2")]

    # ------------------------------------------------------------
    def space(self):
        low = np.full(len(self._metrics), -np.inf, dtype=np.float32)
        high = np.full(len(self._metrics), np.inf, dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def __call__(self):
        agg = dict.fromkeys(self._metrics, 0.0)
        count_occ = 0
        for det_id in self.detectors:
            if "queue_length" in agg:
                agg["queue_length"] += self.traci.lanearea.getJamLengthVehicle(det_id)
            if "queue_length_in_meters" in agg:
                agg["queue_length_in_meters"] += self.traci.lanearea.getJamLengthMeters(det_id)
            if "halt_count" in agg:
                agg["halt_count"] += self.traci.lanearea.getLastStepHaltingNumber(det_id)
            if "occupancy" in agg:
                agg["occupancy"] += self.traci.lanearea.getLastStepOccupancy(det_id)
                count_occ += 1
        if count_occ and "occupancy" in agg:
            agg["occupancy"] /= count_occ
        return agg
