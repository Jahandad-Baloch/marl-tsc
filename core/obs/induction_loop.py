import numpy as np
import gymnasium as gym
from .base import BaseObservationBuilder
from ..registry import register_builder

@register_builder("induction_loop")
class InductionLoopBuilder(BaseObservationBuilder):
    """
    Collects throughput, mean speed, occupancy, queue length,
    etc. from e1/e2 loop detectors attached to this TLS.
    """

    def __init__(self, tls_id, cfg, traci_if, logger=None):
        super().__init__(tls_id, cfg, traci_if, logger)
        m = cfg["metrics"]["obs_metrics"]
        self._metrics = m
        self._init_detectors()

    def _init_detectors(self):
        lanes = {lnk[0] for links in self.traci.trafficlight.getControlledLinks(self.id)
                          for lnk in links}
        pref = self.cfg["detectors"]["detector_prefix"]
        enabled = self.cfg["detectors"]["enabled"]
        self.detectors = [
            f"{pref[tp]}_{ln}" for tp in enabled for ln in lanes
        ]

    # ---- API impl --------------------------------------------------------- #
    def space(self):
        # here we pick fixed ordering of features
        low, high = [], []
        for f in self._metrics:
            low.append(-np.inf)
            high.append(np.inf)
        return gym.spaces.Box(low=np.array(low, dtype=np.float32),
                              high=np.array(high, dtype=np.float32),
                              dtype=np.float32)

    def __call__(self):
        agg, ms_cnt, occ_cnt = dict.fromkeys(self._metrics, 0.0), 0, 0
        traci = self.traci
        for det_id in self.detectors:
            if det_id.startswith("e1det_"):
                if "vehicle_count" in agg:
                    agg["vehicle_count"] += traci.inductionloop.getLastStepVehicleNumber(det_id)
                if "mean_speed" in agg:
                    spd = traci.inductionloop.getLastStepMeanSpeed(det_id)
                    if spd >= 0:
                        agg["mean_speed"] += spd;  ms_cnt += 1
                if "occupancy" in agg:
                    agg["occupancy"] += traci.inductionloop.getLastStepOccupancy(det_id);  occ_cnt += 1
            elif det_id.startswith("e2det_"):
                if "queue_length" in agg:
                    agg["queue_length"] += traci.lanearea.getJamLengthVehicle(det_id)
        if ms_cnt:
            agg["mean_speed"] /= ms_cnt
        if occ_cnt:
            agg["occupancy"] /= occ_cnt
        return agg
