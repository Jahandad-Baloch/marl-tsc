# marl/trainer.py
# This file contains the implementation of the MARL training loop.

import random
import numpy as np
import torch
import torch.nn as nn
# from marl_tsc.advanced_techniques.techniques import apply_shared_rewards, apply_observation_normalization
# /home/jahan_baloch/VSCode_WSL/LibSignal_Evaluations/marl_tsc/advanced_techniques/techniques.py # No Module named 'marl_tsc'
# To resolve this error, we need to import the apply_shared_rewards and apply_observation_normalization functions from the techniques module in the advanced_techniques package.

from marl_tsc.advanced_techniques.techniques import apply_shared_rewards, apply_observation_normalization

from marl_tsc.environment.tsc_env import TrafficSignalControlEnv
from marl_tsc.models.critic import CriticNetwork
from marl_tsc.utils.logger_setup import setup_logger
from marl_tsc.utils.other_utils import set_random_seeds, RunningMeanStd
from marl_tsc.utils.replay_buffer import ReplayBufferCreator
from marl_tsc.agents.agent_factory import create_agent
from marl_tsc.algorithms.algorithm_factory import create_algorithm
import wandb
import neptune


class Trainer:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(config)
        self.init_monitoring()
        set_random_seeds(config['training']['seed'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_environment()
        self.initialize_agents()
        self.initialize_algorithm()
        self.apply_advanced_techniques()

    def init_environment(self):
        self.env = TrafficSignalControlEnv(self.config, logger=self.logger)
        if self.config['training']['register_env']:
            self.env = self.env.register()

        if self.config['training']['normalize_obs']:
            obs_space = self.env.observation_spaces[self.agent_ids[0]]
            obs_dim = obs_space.shape[0]
            self.obs_rms = RunningMeanStd(shape=(obs_dim,))
            self.ret_rms = RunningMeanStd(shape=())

    def initialize_algorithm(self):
        algorithm_name = self.config['model']['algorithm']
        self.algorithm = create_algorithm(algorithm_name, self.config, self.agents, self.env, self.device)

    def initialize_agents(self):
        algorithm = self.config['model']['algorithm']
        self.agents = self.env.agents
        self.agent_ids = self.env.agent_ids
        self.num_agents = len(self.agents)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if algorithm in ['MAPPO', 'PPO']:
            # Initialize actor networks
            self.actor_nets = nn.ModuleDict()
            for agent in self.agents:
                obs_dim = self.env.observation_spaces[agent.id].shape[0]
                action_dim = self.env.action_spaces[agent.id].n
                self.actor_nets[agent.id] = create_agent(self.config, agent.id, obs_dim, action_dim).to(self.device)

        # Initialize centralized critic
        if self.config['model']['use_centralized_critic']:
            observation_spaces = [self.env.observation_spaces[agent.id] for agent in self.agents]
            state_dim = sum([space.shape[0] for space in observation_spaces])
            self.critic = CriticNetwork(state_dim).to(self.device)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config['training']['learning_rate'])

    def init_monitoring(self):
        if self.config.get('use_wandb', False):
            import wandb
            wandb.init(project=self.config['project'])
            wandb.config.update(self.config)
            self.monitor = wandb
        elif self.config.get('use_neptune', False):
            import neptune.new as neptune
            self.run = neptune.init(project=self.config['project'])
            self.run['config'] = self.config
            self.monitor = self.run
        else:
            self.monitor = None


    def train(self):
        num_episodes = self.config['training']['num_episodes']
        for episode in range(num_episodes):
            observations = self.env.reset()
            done = False
            episode_rewards = []
            
            while not done:
                trajectories, observations, done = self.algorithm.collect_rollouts(observations)
                if len(trajectories) == 0:
                    break
                self.algorithm.update(trajectories)
                episode_rewards.extend([sum(t['rewards'].values()) for t in trajectories])

            total_reward = sum(episode_rewards)
            if episode % self.config['logging']['log_interval'] == 0:
                self.logger.info(f"Episode: {episode}, Total Reward: {total_reward}")

    def evaluate(self, num_episodes=10):
        total_rewards = []
        for episode in range(num_episodes):
            observations = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                actions = {}
                for agent_id, agent in self.agents.items():
                    obs = observations[agent_id]
                    if agent.obs_rms:
                        obs = (obs - agent.obs_rms.mean) / (agent.obs_rms.var + 1e-8)
                    action, _ = agent.select_action(obs, evaluate=True)
                    actions[agent_id] = action

                observations, rewards, done, _ = self.env.step(actions)
                episode_reward += sum(rewards.values())

            total_rewards.append(episode_reward)

        average_reward = sum(total_rewards) / num_episodes
        self.logger.info(f"Evaluation over {num_episodes} episodes: Average Reward: {average_reward}")
        return average_reward

    def save_model(self, path):
        state = {
            'agents': {agent_id: agent.get_state_dict() for agent_id, agent in self.agents.items()},
            'algorithm': self.algorithm.get_state_dict(),
            'config': self.config
        }
        torch.save(state, path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path):
        state = torch.load(path)
        for agent_id, agent_state in state['agents'].items():
            self.agents[agent_id].load_state_dict(agent_state)
        self.algorithm.load_state_dict(state['algorithm'])
        self.config = state['config']
        self.logger.info(f"Model loaded from {path}")

    def train_ppo(self):
        num_episodes = self.config['training']['num_episodes']
        for episode in range(num_episodes):
            # Reset the environment at the start of each episode
            observations = self.env.reset()
            self.prev_observations = observations  # Initialize previous observations
            done = False
            episode_rewards = []

            while not done:
                trajectories, observations, done = self.collect_rollouts(observations)
                
                if len(trajectories) == 0:
                    break # End episode if no valid trajectories
                
                # Update normalization
                obs_samples = torch.stack([torch.tensor(obs, dtype=torch.float32) for t in trajectories for obs in t['observations'].values()])
                rewards = torch.tensor([sum(t['rewards'].values()) for t in trajectories], dtype=torch.float32)

                if self.config['training']['normalize_obs']:
                    self.obs_rms.update(obs_samples)
                    self.ret_rms.update(rewards)

                advantages, returns = self.compute_advantages(trajectories)
                self.update_networks(trajectories, advantages, returns)

                episode_rewards.extend([sum(t['rewards'].values()) for t in trajectories])

            total_reward = sum(episode_rewards)
            if episode % self.config['logging']['log_interval'] == 0:
                self.logger.info(f"Episode: {episode}, Total Reward: {total_reward}")
                print(f"Episode: {episode}, Total Reward: {total_reward}")

        self.env.close()
        self.save_model(f"{self.config['logging']['model_dir']}/tscmarl_model.pth")
        self.logger.info("Training complete.")



    def collect_rollouts(self, observations):
        trajectories = []
        done = False

        for step in range(self.config['training']['rollout_length']):
            actions = {}
            action_log_probs = {}
            for agent_id, agent in self.agents.items():
                obs = torch.tensor(observations[agent_id], dtype=torch.float32).to(agent.device)
                action, log_prob = self.actor_nets[agent_id].select_action(obs)
                actions[agent_id] = action
                action_log_probs[agent_id] = log_prob

            # actions, action_log_probs = self.select_actions(observations)
            next_observations, rewards, done, _ = self.env.step(actions)

            if done:
                # Simulation has ended, no valid observations or rewards
                break

            trajectories.append({
                'observations': observations,
                'actions': actions,
                'rewards': rewards,
                'next_observations': next_observations,
                'action_log_probs': action_log_probs,
                'dones': done
            })
            observations = next_observations

        self.prev_observations = observations  # Store observations for the next rollout
        return trajectories, observations, done




