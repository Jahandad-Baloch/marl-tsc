# Description: Implementation of Multi-Agent Proximal Policy Optimization (MAPPO) algorithm
# marl_tsc/algorithms/mappo.py

import torch
import torch.nn as nn
import numpy as np
from marl_tsc.agents.mappo_agent import MAPPOAgent
from marl_tsc.models.actor_critic import CriticNetwork


class MAPPO:
    def __init__(self, config, env, logger=None): 
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.env = env
        self.agent_ids = env.agent_ids
        self.num_agents = len(self.agent_ids)
        
        hidden_dim = config['model']['hidden_dim']

        # Initialize agents
        self.agents = nn.ModuleDict()
        for agent_id in self.agent_ids:
            obs_dim = self.env.observation_spaces[agent_id].shape[0]
            action_dim = self.env.action_spaces[agent_id].n
            continuous_dims = self.env.get_continuous_dims(agent_id)
            self.agents[agent_id] = MAPPOAgent(
                config, agent_id, obs_dim, action_dim, hidden_dim, self.device, continuous_dims=continuous_dims
            )
        # Initialize critic network
        state_dim = sum([space.shape[0] for space in self.env.observation_spaces.values()])

        self.critic = CriticNetwork(state_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config['training']['learning_rate'])

    def collect_rollouts(self, observations):
        trajectories = []
        done = False

        for step in range(self.config['training']['rollout_length']):
            normalized_observations = {}
            for agent_id, agent in self.agents.items():
                obs = observations[agent_id]
                if agent.obs_rms:
                    # Separate continuous and discrete components
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                    obs_continuous = obs_tensor[agent.continuous_dims]
                    # Update running statistics
                    agent.obs_rms.update(obs_continuous)
                    # Normalize continuous components
                    epsilon = self.config['training'].get('normalization_epsilon', 1e-8)
                    obs_continuous = (obs_continuous - agent.obs_rms.mean) / (torch.sqrt(agent.obs_rms.var + epsilon))
                    # Reconstruct normalized observation
                    obs_normalized = obs_tensor.clone()
                    obs_normalized[agent.continuous_dims] = obs_continuous
                    normalized_observations[agent_id] = obs_normalized
                else:
                    normalized_observations[agent_id] = torch.tensor(obs, dtype=torch.float32)

            # Select actions
            actions = {}
            action_log_probs = {}
            for agent_id, agent in self.agents.items():
                obs = normalized_observations[agent_id]
                action, log_prob = agent.select_action(obs)
                actions[agent_id] = action
                action_log_probs[agent_id] = log_prob

            # Step the environment
            next_observations, rewards, done, _ = self.env.step(actions)
            # Store trajectory data
            trajectories.append({
                'observations': observations,
                'normalized_observations': normalized_observations,
                'actions': actions,
                'rewards': rewards,
                'next_observations': next_observations,
                'log_probs': action_log_probs,
                'dones': done
            })
            observations = next_observations

        return trajectories, observations, done



    def collect_rollouts_random_actions(self, observations):
        trajectories = []
        done = False

        for step in range(self.config['training']['rollout_length']):
            actions = {}
            action_log_probs = {}
            for agent_id, agent in self.agents.items():
                obs = observations[agent_id]
                if agent.obs_rms:
                    obs = (obs - agent.obs_rms.mean) / (agent.obs_rms.var + 1e-8)
                action = np.random.choice(self.env.action_spaces[agent_id].n)
                actions[agent_id] = action
                action_log_probs[agent_id] = 0

    def compute_advantages(self, trajectories):
        advantages = {agent_id: [] for agent_id in self.agents}
        returns = {agent_id: [] for agent_id in self.agents}
        gae = 0
        with torch.no_grad():
            values = []
            for idx, trajectory in enumerate(trajectories):
                # Compute value for current state
                obs = [trajectory['observations'][agent_id] for agent_id in self.agent_ids]
                state = torch.tensor(np.concatenate(obs), dtype=torch.float32).unsqueeze(0).to(self.device)
                value = self.critic(state).item()
                values.append(value)
            
            for idx in reversed(range(len(trajectories))):
                reward = sum(trajectories[idx]['rewards'].values())
                if idx < len(trajectories) - 1:
                    next_value = values[idx + 1]
                else:
                    next_value = 0  # Terminal state
                delta = reward + self.config['training']['gamma'] * next_value - values[idx]
                gae = delta + self.config['training']['gamma'] * self.config['training']['gae_lambda'] * gae
                for agent_id in self.agent_ids:
                    advantages[agent_id].insert(0, gae)
                    returns[agent_id].insert(0, gae + values[idx])
        return advantages, returns

    def update(self, trajectories):
        advantages, returns = self.compute_advantages(trajectories)
        # Update actor networks
        for agent_id, agent in self.agents.items():
            # Prepare data
            observations = []
            actions = []
            old_log_probs = []
            adv = advantages[agent_id]
            ret = returns[agent_id]
            for idx, trajectory in enumerate(trajectories):
                observations.append(trajectory['observations'][agent_id])
                actions.append(trajectory['actions'][agent_id])
                old_log_probs.append(trajectory['log_probs'][agent_id])

            # Convert to tensors
            observations = np.array(observations, dtype=np.float32)  # Optimized
            observations = torch.tensor(observations, dtype=torch.float32).to(agent.device)
            actions = torch.tensor(actions).to(agent.device)
            old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(agent.device)
            advantages_agent = torch.tensor(adv, dtype=torch.float32).to(agent.device)
            returns_agent = torch.tensor(ret, dtype=torch.float32).to(agent.device)

            # Update actor
            agent.update(observations, actions, old_log_probs, advantages_agent)

        # Update centralized critic
        # Prepare global observations and returns
        global_observations = []
        returns_all = []
        for idx, trajectory in enumerate(trajectories):
            obs = []
            for agent_id in self.agents:
                obs.append(trajectory['observations'][agent_id])
            global_obs = np.concatenate(obs)
            global_observations.append(global_obs)
            # Assuming returns are similar across agents since the critic is centralized
            returns_all.append(returns[agent_id][idx])
        
        global_observations = np.array(global_observations, dtype=np.float32)
        global_observations = torch.tensor(global_observations, dtype=torch.float32).to(self.device)
        returns_all = torch.tensor(returns_all, dtype=torch.float32).to(self.device)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        values = self.critic(global_observations)
        critic_loss = nn.MSELoss()(values.squeeze(), returns_all)
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()
        


    def get_state_dict(self):
        return {
            'agents': {agent_id: agent.state_dict() for agent_id, agent in self.agents.items()},
            'critic': self.critic.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        for agent_id, agent_state in state_dict['agents'].items():
            self.agents[agent_id].load_state_dict(agent_state)
        self.critic.load_state_dict(state_dict['critic'])
    
    def save_model(self, path):
        state_dict = self.get_state_dict()
        torch.save(state_dict, path)
        if self.logger:
            self.logger.info(f"Model saved at {path}")
        
    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
        if self.logger:
            self.logger.info(f"Model loaded from {path}")    