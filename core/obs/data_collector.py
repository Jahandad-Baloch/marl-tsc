import json

""" 
Description: This script contains the implementation of the DataCollector class, which is responsible for collecting data from the SUMO traffic simulation.
path: TSCMARL/marl_setup/data_collection.py
"""

class DataCollector:
    def __init__(self, tls_id, config, sumo_interface, logger=None):
        self.id = tls_id
        self.config = config
        self.logger = logger
        self.sumo_interface = sumo_interface
        self.detectors = self.initialize_detectors()
        self.feature_means = {}
        self.feature_stds = {}
        self.metrics = self.config.get('metrics', {})
        self.obs_metrics = self.metrics.get('obs_metrics', [])
        self.action_type = self.metrics['action_type']
        self.detector_config = self.config.get('detectors', {})

        detectors = []
        controlled_links = self.sumo_interface.trafficlight.getControlledLinks(self.id)
        in_lanes = {link[0] for links in controlled_links for link in links}
        self.detectors_enabled = self.detector_config.get('enabled', [])

        for detector_type in self.detectors_enabled:
            detector_prefix = self.detector_config['detector_prefix'][detector_type]
            detectors.extend(f"{detector_prefix}_{lane}" for lane in in_lanes)
        return detectors

    def collect_traffic_metrics(self):
        """
        Collect data from enabled detectors, return a dictionary of the collected data.
        """
        aggregated_data = {metric: 0.0 for metric in self.obs_metrics}
        mean_speed_count = 0
        occupancy_count = 0

        for detector_id in self.detectors:
            if detector_id.startswith('e1det_'):

                if 'vehicle_count' in self.obs_metrics:
                    throughput = self.sumo_interface.inductionloop.getLastStepVehicleNumber(detector_id)
                    aggregated_data['throughput'] += throughput
                if 'mean_speed' in self.obs_metrics:
                    mean_speed = self.sumo_interface.inductionloop.getLastStepMeanSpeed(detector_id)
                    if mean_speed >= 0:
                        aggregated_data['mean_speed'] += mean_speed
                        mean_speed_count += 1
                if 'occupancy' in self.obs_metrics:
                    occupancy = self.sumo_interface.inductionloop.getLastStepOccupancy(detector_id)
                    aggregated_data['occupancy'] += occupancy
                    occupancy_count += 1
                if 'waiting_time' in self.obs_metrics:
                    vehicle_ids = self.sumo_interface.inductionloop.getLastStepVehicleIDs(detector_id)
                    vehicle_data = self.sumo_interface.inductionloop.getVehicleData(detector_id)
                    for veh_id, veh_length, entry_time, exit_time, v_type in vehicle_data:
                        if veh_id in vehicle_ids:
                            waiting_time = exit_time - entry_time
                            aggregated_data['waiting_time'] += waiting_time

            elif detector_id.startswith('e2det_'):
                # Lane Area Detectors (Incoming Lanes)
                if 'queue_length' in self.obs_metrics:
                    jam_length = self.sumo_interface.lanearea.getJamLengthVehicle(detector_id)
                    aggregated_data['queue_length'] += jam_length

                if 'queue_length_in_meters' in self.obs_metrics:
                    jam_length_meters = self.sumo_interface.lanearea.getJamLengthMeters(detector_id)
                    aggregated_data['queue_length_in_meters'] += jam_length_meters

                if 'halt_count' in self.obs_metrics:
                    halt_count = self.sumo_interface.lanearea.getLastStepHaltingNumber(detector_id)
                    aggregated_data['halt_count'] += halt_count

        # Calculate averages if counts are greater than zero
        if mean_speed_count > 0:
            aggregated_data['mean_speed'] /= mean_speed_count
        if occupancy_count > 0:
            aggregated_data['occupancy'] /= occupancy_count

        return aggregated_data


    def initialize_detectors(self):
        """
        Initialize detectors based on the configuration.
        """
        detectors = []
        controlled_links = self.sumo_interface.trafficlight.getControlledLinks(self.id)
        in_lanes = {link[0] for links in controlled_links for link in links}
        self.detectors_enabled = self.detector_config.get('enabled', [])

        for detector_type in self.detectors_enabled:
            detector_prefix = self.detector_config['detector_prefix'][detector_type]
            detectors.extend(f"{detector_prefix}_{lane}" for lane in in_lanes)
        return detectors
