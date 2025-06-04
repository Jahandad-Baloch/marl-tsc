import torch.nn as nn

class DQNAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, emb_dim):
        super(DQNAgent, self).__init__()
        self.embedding = nn.Linear(input_dim, emb_dim)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values

    def get_embedding_and_q_values(self, x):
        x = self.embedding(x)              # Embedding layer
        embeddings = x                     # Save embeddings
        x = self.relu(self.fc1(x))
        q_values = self.fc2(x)
        return embeddings, q_values
