import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import random
from marl_tsc.marl.environment.marl_environment import TrafficSignalControlEnv
from marl_tsc.marl.models.actor_critic import ActorNetwork, CriticNetwork
from marl_tsc.marl.utils.replay_buffer import ReplayBuffer
from marl_tsc.marl.utils.config_loader import ConfigLoader
from marl_tsc.marl.utils.logger_setup import LoggerSetup
from marl_tsc.marl.utils.other_utils import ModelParams, TrainingParams, RunningMeanStd, FileIO
import wandb
import neptune
from neptune import Run
""" 
Description: This script contains the implementation of the TSCMARL class, which is responsible for initializing, training, and saving the Traffic Signal Control Multi-Agent Reinforcement Learning (TSCMARL) model.
path: TSCMARL/marl_setup/marl_ctde2.py
"""


class TSCMARL(nn.Module):
    """ 
    Traffic Signal Control Multi-Agent Reinforcement Learning (TSCMARL) model

    Args:
        config_path (str): Path to the configuration file
        logger (Logger): Logger object for logging messages
    """
    def __init__(self, config_path, logger=None):
        super(TSCMARL, self).__init__()
        self.config = ConfigLoader.load_config(config_path)
        self.logger = LoggerSetup.setup_logger(
            self.__class__.__name__.lower(),
            self.config['logging']['log_dir'],
            self.config['logging']['log_level']
        )

        self.model_params = ModelParams(self.config)
        self.training_params = TrainingParams(self.config)
        self.env = TrafficSignalControlEnv(self.config, logger=self.logger)
        self.replay_buffer = ReplayBuffer(
            self.training_params.buffer_size,
            self.training_params.batch_size)


        # Initialize policy and value networks
        self.initialize_models()

    def initialize_models(self):
        self.agents = self.env.agents
        self.agent_ids = self.env.agent_ids
        self.num_agents = len(self.agents)
        self.device = self.training_params.device

        obs_space = self.env.observation_spaces[self.agent_ids[0]]
        obs_dim = obs_space.shape[0]
        # print(f"Observation space: {obs_space}")
        # print(f"Observation dimension: {obs_dim}")
        action_dim = len(self.agents[0].duration_adjustments)
        # print(f"Action dimension: {action_dim}")
        self.actor_nets = nn.ModuleDict()
        for agent in self.agents:
            self.actor_nets[agent.id] = ActorNetwork(obs_dim, action_dim).to(self.device)
        
        state_dim = obs_dim * self.num_agents
        # print(f"State dimension: {state_dim}")
        self.critic = CriticNetwork(state_dim).to(self.device)

        # print first agent's actor network
        # print("First actor network initialization", self.actor_nets[self.agent_ids[0]])
        # print("Centralized critic", self.critic)

        # Optimizers
        self.actor_optimizers = {agent.id: Adam(self.actor_nets[agent.id].parameters(), lr=self.training_params.learning_rate) for agent in self.agents}
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.training_params.learning_rate)

        # Initialize normalization utilities
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.ret_rms = RunningMeanStd(shape=())

        self.gamma = self.training_params.gamma
        self.lam = self.training_params.gae_lambda


        if self.logger:
            self.logger.info(f"Initialized TSCMARL model with {self.num_agents} agents")


    def select_actions(self, observations):
        actions = {}
        action_log_probs = {}
        for agent_id, obs in observations.items():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            # Normalize observation
            obs_norm = (obs_tensor - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)
            action_probs = self.actor_nets[agent_id](obs_norm)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            actions[agent_id] = action.item()
            action_log_probs[agent_id] = dist.log_prob(action).detach()  # Detach here
        return actions, action_log_probs

    def compute_advantages(self, trajectories):

        if len(trajectories) == 0:
            return {}, {}

        advantages = {agent_id: [] for agent_id in self.agent_ids}
        returns = {agent_id: [] for agent_id in self.agent_ids}
        gae = {agent_id: 0 for agent_id in self.agent_ids}
        with torch.no_grad():
            for t in reversed(range(len(trajectories))):
                # Concatenate observations of all agents
                obs_all_agents = torch.cat(
                    [torch.tensor(trajectories[t]['observations'][agent_id], dtype=torch.float32)
                    for agent_id in self.agent_ids], dim=0
                ).to(self.device)
                obs_all_agents = obs_all_agents.unsqueeze(0)  # Shape: (1, state_dim)
                
                if t == len(trajectories) - 1:
                    next_value = 0
                else:
                    next_obs_all_agents = torch.cat(
                        [torch.tensor(trajectories[t+1]['observations'][agent_id], dtype=torch.float32)
                        for agent_id in self.agent_ids], dim=0
                    ).to(self.device)
                    next_obs_all_agents = next_obs_all_agents.unsqueeze(0)
                    next_value = self.critic(next_obs_all_agents).item()
                
                value = self.critic(obs_all_agents).item()
                
                for agent_id in self.agent_ids:
                    reward = trajectories[t]['rewards'][agent_id]
                    delta = reward + self.gamma * next_value - value
                    gae[agent_id] = delta + self.gamma * self.lam * gae[agent_id]
                    advantages[agent_id].insert(0, gae[agent_id])
                    returns[agent_id].insert(0, gae[agent_id] + value)
        return advantages, returns


    def update_networks(self, trajectories, advantages, returns):
        for agent_id in self.agent_ids:
            # Collect this agent's data
            observations = []
            actions = []
            old_log_probs = []
            for t in range(len(trajectories)):
                obs = torch.tensor(trajectories[t]['observations'][agent_id], dtype=torch.float32).to(self.device)
                action = torch.tensor(trajectories[t]['actions'][agent_id]).to(self.device)
                log_prob = trajectories[t]['action_log_probs'][agent_id].to(self.device)
                observations.append(obs)
                actions.append(action)
                old_log_probs.append(log_prob)
            observations = torch.stack(observations)
            actions = torch.stack(actions)
            old_log_probs = torch.stack(old_log_probs)
            advantages_agent = torch.tensor(advantages[agent_id], dtype=torch.float32).to(self.device)
            returns_agent = torch.tensor(returns[agent_id], dtype=torch.float32).to(self.device)

            # Normalize advantages
            advantages_agent = (advantages_agent - advantages_agent.mean()) / (advantages_agent.std() + 1e-8)

            # Update actor network for this agent
            for _ in range(self.training_params.ppo_epochs):
                idx = torch.randperm(len(observations))
                for start in range(0, len(observations), self.training_params.batch_size):
                    end = start + self.training_params.batch_size
                    minibatch_idx = idx[start:end]
                    obs_batch = observations[minibatch_idx]
                    action_batch = actions[minibatch_idx]
                    old_log_prob_batch = old_log_probs[minibatch_idx]
                    advantage_batch = advantages_agent[minibatch_idx]

                    # Get current policy's log probabilities
                    action_probs = self.actor_nets[agent_id](obs_batch)
                    dist = torch.distributions.Categorical(action_probs)
                    log_probs = dist.log_prob(action_batch)

                    # Compute ratio
                    ratios = torch.exp(log_probs - old_log_prob_batch)

                    # Compute surrogate loss
                    surr1 = ratios * advantage_batch
                    surr2 = torch.clamp(ratios, 1 - self.training_params.grad_norm_clip, 1 + self.training_params.grad_norm_clip) * advantage_batch
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Update actor
                    self.actor_optimizers[agent_id].zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_nets[agent_id].parameters(), self.training_params.max_grad_norm)
                    self.actor_optimizers[agent_id].step()

        # Update critic network
        # For the critic, we can process all observations together, as it is a centralized critic
        # Prepare inputs for the critic update
        all_observations = []
        all_returns = []
        for t in range(len(trajectories)):
            obs_all_agents = torch.cat(
                [torch.tensor(trajectories[t]['observations'][agent_id], dtype=torch.float32)
                for agent_id in self.agent_ids], dim=0
            ).to(self.device)
            all_observations.append(obs_all_agents)
            total_return = sum(returns[agent_id][t] for agent_id in self.agent_ids)
            all_returns.append(torch.tensor(total_return, dtype=torch.float32).to(self.device))

        all_observations = torch.stack(all_observations)  # Shape: (T, state_dim)
        all_returns = torch.stack(all_returns)  # Shape: (T,)

        # Update the critic network
        self.critic_optimizer.zero_grad()
        values = self.critic(all_observations).squeeze()
        critic_loss = F.mse_loss(values, all_returns)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.training_params.max_grad_norm)
        self.critic_optimizer.step()


    def collect_rollouts(self, observations):
        trajectories = []
        done = False

        for step in range(self.training_params.rollout_steps):
            actions, action_log_probs = self.select_actions(observations)
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



    def train(self):
        for episode in range(self.training_params.num_episodes):
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
                self.obs_rms.update(obs_samples)
                rewards = torch.tensor([sum(t['rewards'].values()) for t in trajectories], dtype=torch.float32)
                self.ret_rms.update(rewards)
                advantages, returns = self.compute_advantages(trajectories)
                self.update_networks(trajectories, advantages, returns)

                episode_rewards.extend([sum(t['rewards'].values()) for t in trajectories])

            total_reward = sum(episode_rewards)
            episode_metrics = self.env.compute_episode_metrics()
            FileIO.store_metrics(self.training_params.metrics_log_file,
                episode, total_reward, episode_metrics
            )
            if episode % self.training_params.log_interval == 0:
                self.logger.info(f"Episode: {episode}, Total Reward: {total_reward}")
                print(f"Episode: {episode}, Total Reward: {total_reward}")

        self.env.close()
        self.save_model(f"{self.config['logging']['model_dir']}/tscmarl_model.pth")
        self.logger.info("Training complete.")


    def save_model(self, path):
        torch.save({
            'actor_nets_state_dict': {agent_id: net.state_dict() for agent_id, net in self.actor_nets.items()},
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': {agent_id: optimizer.state_dict() for agent_id, optimizer in self.actor_optimizers.items()},
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, path)
        self.logger.info(f"Saved model at {path}")
        
    def load_model(self, path):
        checkpoint = torch.load(path)
        for agent_id, net in self.actor_nets.items():
            net.load_state_dict(checkpoint['actor_nets_state_dict'][agent_id])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        for agent_id, optimizer in self.actor_optimizers.items():
            optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'][agent_id])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.logger.info(f"Loaded model from {path}")
        
    def test(self):
        self.load_model(f"{self.config['logging']['model_dir']}/tscmarl_model.pth")
        observations = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            actions, _ = self.select_actions(observations)
            next_observations, rewards, done, _ = self.env.step(actions)
            total_reward += sum(rewards.values())
            observations = next_observations
        self.logger.info(f"Total reward: {total_reward}")
        self.env.simulator.close_sumo()
        
    def random_actions(self):
        observations = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            actions = {agent_id: random.choice(self.agents[0].duration_adjustments) for agent_id in self.agent_ids}
            next_observations, rewards, done, _ = self.env.step(actions)
            total_reward += sum(rewards.values())
            observations = next_observations
        self.logger.info(f"Total reward: {total_reward}")
        self.env.simulator.close_sumo()
        
    def evaluate(self):
        self.test()
        self.random_actions()

if __name__ == '__main__':
    config_path = 'TSCMARL/configurations/main_config.yaml'

    model = TSCMARL(config_path)
    model.train()