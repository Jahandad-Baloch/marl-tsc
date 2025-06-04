import gymnasium as gym
from gymnasium import spaces
import numpy as np
# from sumo_traffic_simulator import SUMOTrafficSimulator
# from traffic_light_controller import TrafficLightController
# from utils import FileIO
from marl_tsc.simulation.simulator import SUMOTrafficSimulator
from marl_tsc.simulation.tlc import TrafficLightController
from marl_tsc.utils.other_utils import FileIO

class TrafficSignalControlEnv(gym.Env):
    """
    Gymnasium environment for multi-agent traffic signal control using SUMO.
    Agents adjust the duration of the current green phase using discrete duration adjustments.
    """
    def __init__(self, config, logger=None):
        super().__init__()
        self.config = config
        self.logger = logger
        self.simulator = SUMOTrafficSimulator(self.config, self.logger)
        self.metrics = self.config.get('metrics', {})
        self.simulator.initialize_sumo()
        self.initialize_environment()

    def initialize_environment(self):
        self.sumo_interface = self.simulator.sumo_interface
        self.agent_ids = self.simulator.traffic_light_ids  # List of agent IDs as strings
        self.agents = {
            agent_id: TrafficLightController(agent_id, self.config, self.sumo_interface, self.logger) 
            for agent_id in self.agent_ids
        }

        # self.traffic_signal_controllers = {
        #     tls_id: TrafficLightController(tls_id, self.config, self.sumo_interface, self.logger)
        #     for tls_id in self.agent_ids
        # }
        # self.agents = [self.traffic_signal_controllers[tls_id] for tls_id in self.agent_ids]

        self.obs_metrics = self.metrics.get('obs_metrics', [])
        self.reward_metric = self.metrics.get('reward_metric', '')
        self.reward_function = self.metrics.get('reward_function', 'difference')

        # Define action and observation spaces
        self.action_spaces = {agent_id: self.get_action_space(agent_id) for agent_id in self.agents.keys()}
        self.observation_spaces = {agent_id: self.get_observation_space(agent_id) for agent_id in self.agents.keys()}

        self.prev_step_obs = None
        self.data_logs = []



    def reset(self):
        """
        Reset the environment for a new episode.
        """
        if self.logger:
            self.logger.info("Resetting environment for a new episode.")
        self.simulator.reset_sumo()
        self.episode_vehicle_data = {}
        self.episode_metrics = {}
        self.data_logs = []

        for agent_id, agent in self.agents.items():
            agent.reset_tl()

        self.prev_step_obs = None
        observations = self.get_observations()
        self.prev_step_obs = observations.copy()
        return observations

    def step(self, actions):
        """
        Take a step in the environment.
        """
        terminated, truncated = self.simulator.advance_simulation_step()
        done = terminated or truncated

        if done:
            # Simulation has ended, return default values
            observations = self.prev_step_obs  # Use previous observations or set to None
            rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
            info = {}
            return observations, rewards, done, info

        # Apply actions to agents
        for agent_id, action in actions.items():
            self.agents[agent_id].pseudo_step(action)

        # Collect observations
        observations = self.get_observations()

        # Compute rewards
        rewards = self.compute_rewards(observations)

        # Update prev_step_obs
        self.prev_step_obs = observations.copy()

        info = {}
        return observations, rewards, done, info


    def get_observations(self):
        """
        Collect observations from all agents.
        Returns a dictionary with agent IDs as keys and observations as values.
        """
        observations = {}
        for agent in self.agents.values():
            data = agent.collect_data()
            obs_continuous = []
            obs_discrete = []
            for metric in self.obs_metrics:
                value = data.get(metric, 0.0)
                if metric == 'current_phase' and self.metrics.get('use_phase_one_hot', False):
                    # One-hot encoded phase is a discrete variable
                    obs_discrete.extend(value.tolist())
                else:
                    if isinstance(value, np.ndarray):
                        obs_continuous.extend(value.tolist())
                    else:
                        obs_continuous.append(float(value))
            # Combine continuous and discrete observations
            obs = np.array(obs_continuous + obs_discrete, dtype=np.float32)
            observations[agent.id] = obs
        return observations

    def get_continuous_dims(self, agent_id):
        """
        Returns the indices of continuous observation components for an agent.
        """
        continuous_dims = []
        index = 0
        for metric in self.obs_metrics:
            if metric == 'current_phase' and self.metrics.get('use_phase_one_hot', False):
                num_phases = len(self.agents[agent_id].phases)
                index += num_phases  # Skip discrete one-hot encoded phases
            else:
                continuous_dims.append(index)
                index += 1
        return continuous_dims


    def compute_rewards(self, observations):
        """
        Compute rewards for all agents based on the specified reward function.
        """
        rewards = {}
        if self.reward_function == 'difference':
            for agent_id in self.agents.keys():
                prev_value = self.prev_step_obs[agent_id][self.obs_metrics.index(self.reward_metric)]
                current_value = observations[agent_id][self.obs_metrics.index(self.reward_metric)]
                reward = -(current_value - prev_value)
                rewards[agent_id] = reward
        elif self.reward_function == 'global':
            total_current = sum(
                observations[agent_id][self.obs_metrics.index(self.reward_metric)] for agent_id in self.agents.keys()
            )
            reward = -total_current
            for agent_id in self.agent_ids:
                rewards[agent_id] = reward
        else:
            raise ValueError(f"Invalid reward function: {self.reward_function}")
        return rewards


    def get_action_space(self, agent_id):
        """
        Define the action space for the agents.
        Agents adjust the duration of the current green phase using discrete adjustments.
        """
        # num_duration_options = len(self.agents[0].duration_adjustments)
        num_duration_options = len(self.agents[agent_id].duration_adjustments)
        return spaces.Discrete(num_duration_options)


    def get_observation_space(self, agent_id):
        """
        Define the observation space for the agents.
        Observation includes specified metrics and optional one-hot encoded current phase.
        """
        num_state_metrics = len(self.obs_metrics)
        num_discrete_elements = 0
        if 'current_phase' in self.obs_metrics and self.metrics.get('use_phase_one_hot', False):
            num_phases = len(self.agents[agent_id].phases)
            num_discrete_elements += num_phases - 1  # Subtract 1 because 'current_phase' is already counted in obs_metrics
        obs_dim = num_state_metrics + num_discrete_elements
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)


    def compute_episode_metrics(self):
        """
        Compute metrics at the end of an episode using collected vehicle data.
        """
        vehicle_data = self.simulator.get_vehicle_data()
        total_travel_time = 0.0
        total_waiting_time = 0.0
        num_vehicles = 0

        for veh_id, data in vehicle_data.items():
            if data['exit_time'] is not None:
                travel_time = data['exit_time'] - data['entry_time']
                total_travel_time += travel_time
                total_waiting_time += data['waiting_time']
                num_vehicles += 1

        if num_vehicles > 0:
            average_travel_time = total_travel_time / num_vehicles
            average_waiting_time = total_waiting_time / num_vehicles
        else:
            average_travel_time = 0.0
            average_waiting_time = 0.0

        vehicle_throughput = num_vehicles

        self.episode_metrics = {
            'average_travel_time': average_travel_time,
            'vehicle_throughput': vehicle_throughput,
            'average_waiting_time': average_waiting_time,
        }

        return self.episode_metrics


    def close(self):
        """
        Close the environment and clean up resources.
        """
        self.simulator.close_sumo()


    def save_data_to_json_file(self, transition_logs_file, phase_info_file):
        """
        Save the phase information and transition logs to JSON files.
        """
        data = {}
        for tls_id, controller in self.traffic_signal_controllers.items():
            data[tls_id] = {
                'phase_info': controller.phase_info,
            }
        # save phase info
        FileIO.save_to_json(phase_info_file, data)
        # save transition logs
        FileIO.save_to_json(transition_logs_file, self.data_logs)
        
