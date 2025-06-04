import numpy as np
import gymnasium as gym
from .base import BaseObservationBuilder
from ..registry import register_builder


@register_builder("multi_entry_exit")
class MultiEntryExitBuilder(BaseObservationBuilder):
    """Combine induction loop and lane area metrics."""

    def __init__(self, tls_id, cfg, traci_if, logger=None):
        super().__init__(tls_id, cfg, traci_if, logger)
        self._metrics = cfg["metrics"]["obs_metrics"]
        self._init_detectors()

    def _init_detectors(self):
        lanes = {lnk[0] for links in self.traci.trafficlight.getControlledLinks(self.id)
                          for lnk in links}
        pref = self.cfg["detectors"]["detector_prefix"]
        enabled = self.cfg["detectors"]["enabled"]
        self.detectors = [f"{pref[tp]}_{ln}" for tp in enabled for ln in lanes]

    def space(self):
        low = np.full(len(self._metrics), -np.inf, dtype=np.float32)
        high = np.full(len(self._metrics), np.inf, dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def __call__(self):
        agg = dict.fromkeys(self._metrics, 0.0)
        traci = self.traci
        count_speed = 0
        for det_id in self.detectors:
            if det_id.startswith("e1det_"):
                if "vehicle_count" in agg:
                    agg["vehicle_count"] += traci.inductionloop.getLastStepVehicleNumber(det_id)
                if "mean_speed" in agg:
                    spd = traci.inductionloop.getLastStepMeanSpeed(det_id)
                    if spd >= 0:
                        agg["mean_speed"] += spd
                        count_speed += 1
                if "occupancy" in agg:
                    agg["occupancy"] += traci.inductionloop.getLastStepOccupancy(det_id)
            elif det_id.startswith("e2det_"):
                if "queue_length" in agg:
                    agg["queue_length"] += traci.lanearea.getJamLengthVehicle(det_id)
                if "halt_count" in agg:
                    agg["halt_count"] += traci.lanearea.getLastStepHaltingNumber(det_id)
        if count_speed and "mean_speed" in agg:
            agg["mean_speed"] /= count_speed
        return agg
