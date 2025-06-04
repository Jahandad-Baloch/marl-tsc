# marl/agents/mappo_agent.py

import torch
from marl_tsc.agents.base_agent import BaseAgent
from marl_tsc.models.actor_critic import ActorNetwork


class MAPPOAgent(BaseAgent):
    def __init__(self, config, agent_id, obs_dim, action_dim, hidden_dim=128, device='cpu', continuous_dims=None):
        super(MAPPOAgent, self).__init__(config, agent_id, obs_dim, continuous_dims)
        self.device = device
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim).to(self.device)

        # Additional initializations
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['training']['learning_rate'])
        self.clip_param = config['training']['clip_param']
        self.max_grad_norm = config['training']['max_grad_norm']
        self.ppo_epochs = config['training']['ppo_epochs']
        self.batch_size = config['training']['batch_size']


    def select_action(self, observation):
        if not isinstance(observation, torch.Tensor):
            obs_tensor = torch.tensor(observation, dtype=torch.float32).to(self.device)
        else:
            obs_tensor = observation.to(self.device)
        with torch.no_grad():
            action_probs = self.actor(obs_tensor)
            # Ensure action_probs are valid probabilities
            action_probs = torch.softmax(action_probs, dim=-1)
            print("Action probs for agent ", self.agent_id, ": ", action_probs)

            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()

    def get_log_prob(self, observation, action):
        obs_tensor = torch.tensor(observation, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(action).to(self.device)
        action_probs = self.actor(obs_tensor)
        dist = torch.distributions.Categorical(action_probs)
        log_prob = dist.log_prob(action_tensor)
        return log_prob

    def update(self, observations, actions, old_log_probs, advantages):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dataset = torch.utils.data.TensorDataset(observations, actions, old_log_probs, advantages)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.ppo_epochs):
            for batch in dataloader:
                batch_obs, batch_actions, batch_old_log_probs, batch_advantages = [x.to(self.device) for x in batch]
                action_probs = self.actor(batch_obs)
                dist = torch.distributions.Categorical(action_probs)
                log_probs = dist.log_prob(batch_actions)
                ratios = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()


    

