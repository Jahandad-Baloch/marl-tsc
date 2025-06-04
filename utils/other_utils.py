# marl_tsc/utils/other_utils.py

import numpy as np 
import json
import os
import yaml
import logging
import torch
from xml.etree import ElementTree as ET
import subprocess
from datetime import datetime
import pandas as pd
import numpy as np
import random
from xml.dom import minidom
import csv




# Class ModelParams to load all the parameters of the model
class ModelParams:
    def __init__(self, config):        

        # Model specific parameters
        model_params = config['model']
        self.agent_network = model_params['agent_network']
        self.algorithm = model_params['algorithm']
        self.exploration_strategy = model_params['exploration_strategy']
        self.replay_buffer_type = model_params['replay_buffer_type']
        if self.replay_buffer_type == 'prioritized':
            self.prioritized_replay = config['prioritized_replay']

        # Model architecture
        self.hidden_dim = model_params['hidden_dim']
        self.hypernet_embed_dim = model_params['hypernet_embed_dim']
        self.neighbor_num = model_params['neighbor_num']
        self.neighbor_edge_num = model_params['neighbor_edge_num']
        self.node_emb_dim = model_params['node_emb_dim']
        self.input_dims = model_params['input_dims']
        self.node_layer_dims_each_head = model_params['node_layer_dims_each_head']
        self.output_dims = model_params['output_dims']
        self.num_heads = model_params['num_heads']
        
        # Additional model flags
        self.one_hot = model_params['one_hot']

def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Class ModelParams to load all the parameters of the model
class TrainingParams:
    def __init__(self, config):

        # Training parameters
        training_params = config['training']
        self.batch_size = training_params['batch_size']
        self.buffer_size = training_params['buffer_size']
        self.replay_buffer_size = training_params['buffer_size']
        self.model_path = training_params['model_path']

        self.update_interval = training_params['update_interval']
        self.update_target_rate = training_params['update_target_rate']
        self.update_model_rate = training_params['update_model_rate']
        self.save_interval = training_params['save_interval']
        self.log_interval = training_params['log_interval']
        self.eval_interval = training_params['eval_interval']

        self.test_when_train = training_params['test_when_train']
        self.start_training = training_params['start_training']
        self.warmup_steps = training_params['warmup_steps']
        self.max_steps = training_params['max_steps']
        self.seed = training_params['seed']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_episodes = training_params['num_episodes']
        self.num_eval_episodes = training_params['num_eval_episodes']
        # self.reward_function = training_params['reward_function']

        self.learning_rate = training_params['learning_rate']
        self.gamma = training_params['gamma']
        self.tau = training_params['tau']
        self.epsilon = training_params['epsilon']
        self.epsilon_decay = training_params['epsilon_decay']
        self.epsilon_min = training_params['epsilon_min']
        self.epsilon_start = training_params['epsilon_start']
        self.replay_eps = training_params['replay_eps']
        self.replay_eps_end = training_params['replay_eps_end']
        self.replay_eps_annealing = training_params['replay_eps_annealing']
        self.grad_norm_clip = training_params['grad_norm_clip'] # Gradient clipping e.g. 5.0
        self.max_grad_norm = training_params['max_grad_norm'] # Maximum gradient norm e.g. 0.5
        self.gae_lambda = training_params['gae_lambda'] # Generalized Advantage Estimation e.g. 0.95
        self.rollout_steps = training_params['rollout_steps'] # Number of steps to unroll the PPO agent e.g. 5
        self.ppo_epochs = training_params['ppo_epochs'] # Number of epochs to train the PPO agent e.g. 4
        self.metrics_log_file = os.path.join(config['logging']['log_dir'], 'metrics_log.csv')


class ConfigLoader:
    @staticmethod
    def load_config(config_path):
        """
        Load configuration from a YAML file.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            dict: Loaded configuration.
        """
        with open(config_path) as file:
            configs = yaml.full_load(file)
            base_path = os.path.dirname(config_path)
            for conf in configs.get('imports', []):
                conf_path = os.path.join(base_path, conf)
                with open(conf_path) as f:
                    configs.update(yaml.full_load(f))
            return configs

class FileIO:
    @staticmethod
    def create_csv(path: str, columns: list, logger):
        df = pd.DataFrame(columns=columns)
        df.to_csv(path, index=False)
        logger.info(f"CSV file created at {path}")

    @staticmethod
    def save_to_csv(data: pd.DataFrame, path: str, logger):
        data.to_csv(path, index=False)
        logger.info(f"Data saved to {path}")

    @staticmethod
    def add_to_csv(data: pd.DataFrame, path: str, logger):
        # To add but not replace the data, use mode='a' and header=False
        data.to_csv(path, mode='a', header=False, index=False)
        logger.info(f"Data added to {path}")
        
    @staticmethod
    def save_to_json(data: dict, path: str, logger):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Data saved to {path}")

    @staticmethod
    def store_metrics(metrics_log_file, episode, total_reward, episode_metrics):
        """
        Store metrics to a CSV file.
        """
        with open(metrics_log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Episode', 'Total Reward', 'Average Travel Time', 'Vehicle Throughput', 'Average Waiting Time']
            writer.writerow(header)
            writer.writerow([
                episode,
                total_reward,
                episode_metrics.get('average_travel_time', 0.0),
                episode_metrics.get('vehicle_throughput', 0),
                episode_metrics.get('average_waiting_time', 0.0),
            ])

        # Placeholder for Neptune logging
        # import neptune.new as neptune
        # run = neptune.init(...)
        # run['metrics/total_reward'].log(total_reward)
        # run['metrics/average_travel_time'].log(episode_metrics['average_travel_time'])
        # run['metrics/vehicle_throughput'].log(episode_metrics['vehicle_throughput'])
        # run['metrics/average_waiting_time'].log(episode_metrics['average_waiting_time'])

class LoggerSetup:
    @staticmethod
    def setup_logger(name, log_dir, level=logging.INFO):
        """
        Setup a logger with the specified name and log directory.

        Args:
            name (str): Name of the logger.
            log_dir (str): Directory to store log files.
            level (int, optional): Logging level. Defaults to logging.INFO.

        Returns:
            logging.Logger: Configured logger.
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%m%d')}.log")
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger



""" 
This script defines the edge types for the SUMO network.
path: scripts/common/edge_types_xml.py
"""

class EdgeTypesXML:
    """
    Class to generate edge types XML for the SUMO network.
    """
    type_mappings = {
        'arterial': {
            201100: {'id': '201100', 'priority': 5, 'numLanes': 4, 'speed': 27.78, 'oneway': 'true', 'name': 'Expressway', 'allow': 'passenger taxi delivery emergency evehicle truck bus coach trailer'},
            201800: {'id': '201800', 'priority': 4, 'numLanes': 1, 'speed': 25.0, 'oneway': 'true', 'name': 'Expressway Ramp', 'allow': 'passenger taxi delivery emergency evehicle truck bus coach trailer'},
            201200: {'id': '201200', 'priority': 4, 'numLanes': 3, 'speed': 12.5, 'oneway': 'true', 'name': 'Major Arterial', 'allow': 'passenger taxi delivery emergency evehicle truck bus coach trailer'},
            201300: {'id': '201300', 'priority': 3, 'numLanes': 2, 'speed': 12.5, 'oneway': 'true', 'name': 'Minor Arterial', 'allow': 'passenger taxi delivery emergency evehicle truck bus coach trailer'},
            201600: {'id': '201600', 'priority': 4, 'numLanes': 1, 'speed': 12.5, 'oneway': 'true', 'name': 'Major Arterial Ramp', 'allow': 'passenger taxi delivery emergency evehicle truck bus coach trailer'},
            201301: {'id': '201301', 'priority': 3, 'numLanes': 1, 'speed': 12.5, 'oneway': 'true', 'name': 'Minor Arterial Ramp', 'allow': 'passenger taxi delivery emergency evehicle truck bus coach trailer'},
            201801: {'id': '201801', 'priority': 2, 'numLanes': 1, 'speed': 11.11, 'oneway': 'false', 'name': 'Busway', 'allow': 'passenger taxi delivery emergency evehicle truck bus coach trailer'}
        },
        'collector': {
            201400: {'id': '201400', 'priority': 3, 'numLanes': 2, 'speed': 12.5, 'oneway': 'false', 'name': 'Collector', 'allow': 'passenger taxi delivery emergency evehicle truck bus coach trailer'},
            201401: {'id': '201401', 'priority': 3, 'numLanes': 1, 'speed': 11.11, 'oneway': 'true', 'name': 'Collector Ramp', 'allow': 'passenger taxi delivery emergency evehicle truck bus coach trailer'},
            201803: {'id': '201803', 'priority': 1, 'numLanes': 1, 'speed': 5.56, 'oneway': 'false', 'name': 'Access Road', 'allow': 'passenger taxi delivery emergency evehicle truck bus coach trailer pedestrian bicycle moped motorcycle scooter'},
            201601: {'id': '201601', 'priority': 4, 'numLanes': 1, 'speed': 12.5, 'oneway': 'true', 'name': 'Other Ramp', 'allow': 'passenger taxi delivery emergency evehicle truck bus coach trailer'},
        },
        'local': {
            201500: {'id': '201500', 'priority': 2, 'numLanes': 2, 'speed': 12.5, 'oneway': 'false', 'name': 'Local', 'allow': 'passenger taxi delivery emergency evehicle truck bus coach trailer'},
            201700: {'id': '201700', 'priority': 1, 'numLanes': 1, 'speed': 4.17, 'oneway': 'false', 'name': 'Laneway', 'allow': 'pedestrian bicycle moped motorcycle scooter passenger taxi emergency evehicle'},
            204001: {'id': '204001', 'priority': 1, 'numLanes': 1, 'speed': 5.56, 'oneway': 'false', 'name': 'Trail', 'allow': 'pedestrian bicycle moped motorcycle scooter'},
            204002: {'id': '204002', 'priority': 1, 'numLanes': 1, 'speed': 5.56, 'oneway': 'false', 'name': 'Walkway', 'allow': 'pedestrian bicycle moped motorcycle scooter'}
        }
    }

    @staticmethod
    def create(network_type, output_file):
        """
        Create an XML representation of edge types from dynamic attributes, reusing arterial lanes.
        Save the XML string to a file.
        Args:
            network_type (str): The type of network (arterial, collector, local).

        Returns:
            dict: type mappings for the network type.
        """
        # Start with arterial types and extend based on the selected network type
        active_mappings = EdgeTypesXML.type_mappings['arterial'].copy()
        
        if network_type == 'collector':
            # Add specific collector lanes
            active_mappings.update(EdgeTypesXML.type_mappings['collector'])
        elif network_type == 'local':
            # Add specific local lanes
            active_mappings.update(EdgeTypesXML.type_mappings['local'])
        
        types = ET.Element("types")
        for feature_code, attributes in active_mappings.items():
            type_element = ET.SubElement(types, "type")
            for attr_key, attr_value in attributes.items():
                type_element.set(attr_key, str(attr_value))

        rough_string = ET.tostring(types, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        xml_str = reparsed.toprettyxml(indent="  ")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(xml_str)
        
        return active_mappings

""" 
This script is used to execute commands.
path: scripts/common/command_executor.py
"""

class CommandExecutor:
    def __init__(self, logger=None):
        """
        Initializes the CommandExecutor.

        Args:
            logger (logging.Logger): Optional logger for logging command output. If None, a default logger will be used.
        """
        self.logger = logger

    def run_command(self, command, shell=False, capture_output=True, check=True, cwd=None, env=None):
        """
        Executes a command using subprocess and logs the output.

        Args:
            command (str or list): Command to execute.
            shell (bool): Whether to use the shell as the program to execute. Defaults to False.
            capture_output (bool): Whether to capture stdout and stderr. Defaults to True.
            check (bool): If True, raise an exception on command failure. Defaults to True.
            cwd (str): If not None, change to this directory before running the command.
            env (dict): Environment variables to use for the command.

        Returns:
            subprocess.CompletedProcess: The result of the executed command.

        Raises:
            subprocess.CalledProcessError: If the command fails and check is True.
        """

        try:
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=capture_output,
                check=check,
                text=True,
                cwd=cwd,
                env=env
            )
            if capture_output:
                self.logger.debug(f"Command stdout: {result.stdout}")
                self.logger.debug(f"Command stderr: {result.stderr}")
            self.logger.info("Command executed successfully.")
            return result
        except subprocess.CalledProcessError as e:
            self.logger.info("Command executed failed.")
            self.logger.error(f"Error executing command: {e}")
            if capture_output:
                self.logger.error(f"Command stdout: {e.stdout}")
                self.logger.error(f"Command stderr: {e.stderr}")
            raise

    def get_sumo_tools_path(self):
        """
        Get the path to the SUMO tools directory.

        Returns:
            str: Path to the SUMO tools directory.
        """
        sumo_home = os.environ.get('SUMO_HOME')
        if sumo_home is None:
            raise EnvironmentError("SUMO_HOME environment variable not set.")
        return os.path.join(sumo_home, 'tools') 