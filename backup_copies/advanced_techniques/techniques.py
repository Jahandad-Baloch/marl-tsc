# marl/advanced_techniques/techniques.py
# This file contains the implementation of advanced techniques for MARL training.

import torch
# marl/advanced_techniques/techniques.py

def apply_advanced_techniques(env, agents, techniques_config):
    if techniques_config.get('normalize_observations', False):
        for agent in agents.values():
            agent.enable_observation_normalization()
    if techniques_config.get('shared_rewards', False):
        env.enable_shared_rewards()
    # Add other techniques as needed

def apply_shared_rewards(rewards, config):
    if config['advanced_techniques']['shared_rewards']:
        # Implement shared rewards logic
        global_reward = sum(rewards.values()) / len(rewards)
        shared_rewards = {agent_id: global_reward for agent_id in rewards}
        return shared_rewards
    return rewards

def apply_observation_normalization(observations, running_stats, config):
    if config['advanced_techniques']['observation_normalization']:
        normalized_observations = {}
        for agent_id, obs in observations.items():
            obs = (obs - running_stats.mean) / (running_stats.std + 1e-8)
            normalized_observations[agent_id] = obs
        return normalized_observations
    return observations

class RunningMeanStd:
    def __init__(self, epsilon=1e-5, shape=()):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = epsilon
    
    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.size(0)
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total_count
        new_var = M2 / (total_count)
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count