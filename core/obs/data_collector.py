import json

class DataCollector:
    """Utility class for fetching raw traffic metrics from SUMO."""

    def __init__(self, tls_id, config, sumo_interface, logger=None):
        self.id = tls_id
        self.config = config
        self.logger = logger
        self.sumo_interface = sumo_interface
        self.detector_config = self.config.get('detectors', {})
        self.metrics = self.config.get('metrics', {})
        self.obs_metrics = self.metrics.get('obs_metrics', [])
        self.action_type = self.metrics.get('action_type', '')

        self.detectors = self._init_detectors()

    def _init_detectors(self):
        detectors = []
        controlled_links = self.sumo_interface.trafficlight.getControlledLinks(self.id)
        in_lanes = {link[0] for links in controlled_links for link in links}
        enabled = self.detector_config.get('enabled', [])
        pref = self.detector_config.get('detector_prefix', {})
        for detector_type in enabled:
            prefix = pref[detector_type]
            detectors.extend(f"{prefix}_{lane}" for lane in in_lanes)
        return detectors

    # ------------------------------------------------------------------
    def get_queue_length(self):
        return self.collect_traffic_metrics().get('queue_length', 0.0)

    def get_waiting_time(self):
        return self.collect_traffic_metrics().get('waiting_time', 0.0)

    # Backwards compatibility
    def get_wait_time(self):
        return self.get_waiting_time()

    def get_emergency_vehicle_count(self):
        return 0

    def get_pedestrian_button_pressed(self):
        return False

    # ------------------------------------------------------------------
    def collect_traffic_metrics(self):
        aggregated_data = {metric: 0.0 for metric in self.obs_metrics}
        mean_speed_count = 0
        occupancy_count = 0

        for detector_id in self.detectors:
            if detector_id.startswith('e1det_'):
                if 'vehicle_count' in self.obs_metrics:
                    aggregated_data['vehicle_count'] += self.sumo_interface.inductionloop.getLastStepVehicleNumber(detector_id)
                if 'mean_speed' in self.obs_metrics:
                    mean_speed = self.sumo_interface.inductionloop.getLastStepMeanSpeed(detector_id)
                    if mean_speed >= 0:
                        aggregated_data['mean_speed'] += mean_speed
                        mean_speed_count += 1
                if 'occupancy' in self.obs_metrics:
                    aggregated_data['occupancy'] += self.sumo_interface.inductionloop.getLastStepOccupancy(detector_id)
                    occupancy_count += 1
                if 'waiting_time' in self.obs_metrics:
                    vehicle_ids = self.sumo_interface.inductionloop.getLastStepVehicleIDs(detector_id)
                    vehicle_data = self.sumo_interface.inductionloop.getVehicleData(detector_id)
                    for veh_id, _, entry_time, exit_time, _ in vehicle_data:
                        if veh_id in vehicle_ids:
                            aggregated_data['waiting_time'] += exit_time - entry_time
            elif detector_id.startswith('e2det_'):
                if 'queue_length' in self.obs_metrics:
                    aggregated_data['queue_length'] += self.sumo_interface.lanearea.getJamLengthVehicle(detector_id)
                if 'queue_length_in_meters' in self.obs_metrics:
                    aggregated_data['queue_length_in_meters'] += self.sumo_interface.lanearea.getJamLengthMeters(detector_id)
                if 'halt_count' in self.obs_metrics:
                    aggregated_data['halt_count'] += self.sumo_interface.lanearea.getLastStepHaltingNumber(detector_id)

        if mean_speed_count > 0 and 'mean_speed' in aggregated_data:
            aggregated_data['mean_speed'] /= mean_speed_count
        if occupancy_count > 0 and 'occupancy' in aggregated_data:
            aggregated_data['occupancy'] /= occupancy_count

        return aggregated_data
