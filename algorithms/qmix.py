import torch
import torch.nn as nn
import torch.nn.functional as F


class MixingNetwork(nn.Module):
    """ 
    Hypernetwork for mixing agent Q-values
    """
    def __init__(self, num_agents, state_dim, hypernet_embed_dim):
        super(MixingNetwork, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hypernet_embed_dim = hypernet_embed_dim
        
        # Hypernetworks for mixing weights and biases
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, num_agents * hypernet_embed_dim)
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, hypernet_embed_dim)
        )
        
        self.hyper_b1 = nn.Linear(state_dim, hypernet_embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, 1)
        )
        
    def forward(self, agent_qs, state):
        batch_size = agent_qs.size(0)
        
        # Generate hypernet weights for the first layer
        w1 = torch.abs(self.hyper_w1(state))  # Shape: [batch_size, num_agents * embed_dim]
        w1 = w1.view(batch_size, self.num_agents, self.hypernet_embed_dim)  # Shape: [batch_size, num_agents, embed_dim]
        
        b1 = self.hyper_b1(state).view(batch_size, 1, self.hypernet_embed_dim)  # Shape: [batch_size, 1, embed_dim]
        
        # Reshape agent_qs for batch multiplication
        agent_qs = agent_qs.view(batch_size, 1, self.num_agents)  # Shape: [batch_size, 1, num_agents]
        
        # First layer transformation
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)  # Shape: [batch_size, 1, embed_dim]
        
        # Generate hypernet weights for the second layer
        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.hypernet_embed_dim, 1)  # Shape: [batch_size, embed_dim, 1]
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)  # Shape: [batch_size, 1, 1]
        
        # Second layer transformation
        y = torch.bmm(hidden, w2) + b2  # Shape: [batch_size, 1, 1]
        q_total = y.view(batch_size)  # Shape: [batch_size]
        return q_total

