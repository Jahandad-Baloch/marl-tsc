# traffic_light_controller.py
import numpy as np
# from data_collection import DataCollector
from marl_tsc.core.obs.data_collector import DataCollector
from datetime import datetime
import torch


class TrafficLightController:
    """
    Represents a controllable traffic light junction with a fixed phase sequence.
    Agents can adjust the duration of the current green phase within regulatory limits.
    """
    def __init__(self, tls_id, config, sumo_interface, logger=None):
        self.id = tls_id
        self.config = config
        self.logger = logger
        self.sumo_interface = sumo_interface
        self.metrics = self.config.get('metrics', {})
        self.obs_metrics = self.metrics.get('obs_metrics', [])
        self.duration_adjustments = [-5.0, 0.0, 5.0]  # Duration adjustments for each action in seconds

        self.configure_traffic_light()

    def configure_traffic_light(self):
        self.phases = self.read_phases()
        self.data_collector = DataCollector(self.id, self.config, self.sumo_interface, self.logger)
        self.current_phase_index = 0  # Start from the first phase
        self.setup_phase_durations()
        self.reset_tl()

    def read_phases(self):
        """
        Read the TLS programs and fetch the phases for a given traffic light.
        """
        try:
            logic = self.sumo_interface.trafficlight.getAllProgramLogics(self.id)
            phases = logic[0].phases
            self.phase_info = []
            for idx, phase in enumerate(phases):
                phase_type = self.identify_phase_type(phase.state)
                self.phase_info.append({
                    'index': idx,
                    'duration': phase.duration,
                    'state': phase.state,
                    'type': phase_type
                })
            return phases
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error retrieving phases for traffic light {self.id}: {e}")
            return []

    def identify_phase_type(self, state_str):
        """
        Identify the phase type based on the signal state string.
        """
        # Simplified rules for phase identification
        if 'G' in state_str or 'g' in state_str:
            return 'green'
        elif 'y' in state_str or 'Y' in state_str:
            return 'yellow'
        elif all(c == 'r' for c in state_str):
            return 'red'
        else:
            return 'other'

    def setup_phase_durations(self):
        """
        Set up the minimum and maximum durations for each phase based on regulatory requirements.
        """
        # Regulatory minimum and maximum durations in seconds for each phase type
        self.regulatory_min_durations = {
            'green': 7.0,
            'yellow': 3.0,
            'red': 5.0,
            'other': 3.0
        }
        self.regulatory_max_durations = {
            'green': 60.0,
            'yellow': 5.0,
            'red': 5.0,
            'other': 5.0
        }

        self.phase_min_durations = {}
        self.phase_max_durations = {}
        for phase in self.phase_info:
            phase_type = phase['type']
            min_duration = self.regulatory_min_durations.get(phase_type, 0)
            max_duration = self.regulatory_max_durations.get(phase_type, float('inf'))
            self.phase_min_durations[phase['index']] = min_duration
            self.phase_max_durations[phase['index']] = max_duration

    def reset_tl(self):
        """
        Reset the traffic light to its initial state.
        """
        self.current_phase_index = 0  # Start from the first phase
        self.current_phase = self.phase_info[self.current_phase_index]['index']
        self.current_phase_start_time = self.sumo_interface.simulation.getTime()
        self.current_phase_duration = self.phase_info[self.current_phase_index]['duration']
        self.sumo_interface.trafficlight.setPhase(self.id, self.current_phase)
        self.transition_log = []
        # Log the initial phase
        self.log_event({
            'current_time': self.current_phase_start_time,
            'previous_phase': None,
            'new_phase': self.current_phase,
            'actual_duration': 0,
            'min_duration': self.phase_min_durations[self.current_phase],
            'max_duration': self.phase_max_durations[self.current_phase],
            'transition_reason': 'initialization'
        })

    def pseudo_step(self, action):
        """
        Adjust the duration of the current green phase based on the action.
        """
        try:
            current_time = self.sumo_interface.simulation.getTime()
            elapsed_time = current_time - self.current_phase_start_time
            current_phase_type = self.phase_info[self.current_phase_index]['type']

            min_duration = self.phase_min_durations.get(self.current_phase, 0)
            max_duration = self.phase_max_durations.get(self.current_phase, float('inf'))

            # Only adjust duration if the current phase is green
            if current_phase_type == 'green' and not hasattr(self, 'duration_adjusted'):
                # Map action index to duration adjustment
                duration_adjustment = self.duration_adjustments[action]
                # Calculate new duration
                adjusted_duration = self.current_phase_duration + duration_adjustment
                # Ensure new duration is within limits
                adjusted_duration = max(min_duration, adjusted_duration)
                adjusted_duration = min(max_duration, adjusted_duration)
                self.current_phase_duration = adjusted_duration
                self.duration_adjusted = True  # Ensure adjustment happens only once per phase

            # Check if it's time to transition to the next phase
            if elapsed_time >= self.current_phase_duration:
                self.duration_adjusted = False  # Reset for the next green phase
                # Transition to the next phase in the sequence
                self.transition_to_next_phase()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error executing action for traffic light {self.id}: {e}")
            raise

    def transition_to_next_phase(self):
        """
        Transition to the next phase in the fixed sequence.
        """
        current_time = self.sumo_interface.simulation.getTime()
        prev_phase = self.current_phase

        # Advance to the next phase index
        self.current_phase_index = (self.current_phase_index + 1) % len(self.phase_info)
        self.current_phase = self.phase_info[self.current_phase_index]['index']
        self.current_phase_start_time = current_time
        self.current_phase_duration = self.phase_info[self.current_phase_index]['duration']

        # Ensure phase duration is within regulatory limits
        min_duration = self.phase_min_durations.get(self.current_phase, 0)
        max_duration = self.phase_max_durations.get(self.current_phase, float('inf'))
        self.current_phase_duration = max(min_duration, self.current_phase_duration)
        self.current_phase_duration = min(max_duration, self.current_phase_duration)

        # Set the new phase
        self.sumo_interface.trafficlight.setPhase(self.id, self.current_phase)

        # Log the phase change
        self.log_event({
            'current_time': current_time,
            'previous_phase': prev_phase,
            'new_phase': self.current_phase,
            'actual_duration': current_time - self.current_phase_start_time,
            'min_duration': min_duration,
            'max_duration': max_duration,
            'transition_reason': 'scheduled'
        })

    def log_event(self, data):
        """
        Log an event in the traffic light controller's data log.
        """
        # Ensure all data values are in a serializable format
        serializable_data = {key: (int(value) if isinstance(value, np.integer) else
                                   float(value) if isinstance(value, np.floating) else
                                   value)
                             for key, value in data.items()}
        event_data = {
            "timestamp": datetime.now().strftime("%H:%M:%S.%f"),  # To include milliseconds
            "TL ID": self.id,
            "data": serializable_data
        }
        self.transition_log.append(event_data)

    
    def collect_data(self):
        """
        Collect data from the traffic light controller for use in the agent's observation space.
        """
        data = self.data_collector.collect_traffic_metrics()
        if 'current_phase' in self.obs_metrics:
            if self.metrics.get('use_phase_one_hot', False):
                current_phase = self.get_current_phase_one_hot()
                data['current_phase'] = current_phase
            else:
                data['current_phase'] = float(self.current_phase)
        return data

    
    def get_current_phase_one_hot(self):
        """
        Return the current phase as a one-hot encoded vector.
        """
        self.current_phase = self.sumo_interface.trafficlight.getPhase(self.id)
        phase_index = self.current_phase
        num_phases = len(self.phases)
        one_hot = np.zeros(num_phases, dtype=float)  # Use float dtype for PyTorch compatibility
        one_hot[phase_index] = 1.0
        return one_hot

    def should_transition_phase(self):
        """
        Check if the current phase duration has elapsed.
        """
        current_time = self.sumo_interface.simulation.getTime()
        elapsed_time = current_time - self.current_phase_start_time
        return elapsed_time >= self.current_phase_duration
