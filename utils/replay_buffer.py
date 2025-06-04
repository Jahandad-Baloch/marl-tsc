import random
import threading
import queue
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []
        self.position = 0
    
    def __len__(self):
        return len(self.buffer)

    def push(self, observations, actions, reward, next_observations, done):
        if not next_observations or len(next_observations) == 0:
            # next_observations is empty, fill with zeros
            next_observations = {agent_id: torch.zeros_like(observations[agent_id]) for agent_id in observations.keys()}
        transition = {
            'observations': observations,
            'actions': actions,
            'reward': reward,
            'next_observations': next_observations,
            'done': done
        }

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.buffer_size


    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)

        # Initialize batches
        batch_observations = {agent_id: [] for agent_id in self.buffer[0]['observations'].keys()}
        batch_next_observations = {agent_id: [] for agent_id in self.buffer[0]['observations'].keys()}
        batch_actions = {agent_id: [] for agent_id in self.buffer[0]['actions'].keys()}
        batch_rewards = []
        batch_dones = []

        for transition in batch:
            for agent_id in transition['observations'].keys():
                batch_observations[agent_id].append(transition['observations'][agent_id])
                batch_next_observations[agent_id].append(transition['next_observations'][agent_id])
                batch_actions[agent_id].append(transition['actions'][agent_id])
            batch_rewards.append(transition['reward'])
            batch_dones.append(transition['done'])

        # Convert lists to tensors
        for agent_id in batch_observations.keys():
            batch_observations[agent_id] = torch.stack(batch_observations[agent_id])
            batch_next_observations[agent_id] = torch.stack(batch_next_observations[agent_id])

            # Handle actions
            actions = batch_actions[agent_id]
            if isinstance(actions[0], (tuple, list)):
                # For MultiDiscrete action spaces
                batch_actions[agent_id] = torch.tensor(actions, dtype=torch.long)
            else:
                batch_actions[agent_id] = torch.tensor(actions, dtype=torch.long).unsqueeze(-1)

        # Convert rewards and dones to tensors
        if isinstance(batch_rewards[0], dict):
            # If rewards are per-agent dictionaries
            agent_ids = batch_rewards[0].keys()
            rewards = {agent_id: torch.tensor([r[agent_id] for r in batch_rewards], dtype=torch.float32) for agent_id in agent_ids}
        else:
            # If rewards are scalars
            rewards = torch.tensor(batch_rewards, dtype=torch.float32)

        dones = torch.tensor(batch_dones, dtype=torch.float32)

        # Return a dictionary
        return {
            'obs': batch_observations,
            'actions': batch_actions,
            'rewards': rewards,
            'next_obs': batch_next_observations,
            'dones': dones
        }



class AsyncReplayBuffer(ReplayBuffer):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.queue = queue.Queue(maxsize=100)  # Adjust maxsize as needed
        self.loading_thread = threading.Thread(target=self._async_load, daemon=True)
        self.loading_thread.start()

    def push(self, *args):
        self.queue.put(args)

    def _async_load(self):
        while True:
            experience = self.queue.get()
            super().push(*experience)
            self.queue.task_done()

class PrioritizedReplayBuffer:
    def __init__(self, capacity, batch_size, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, *experience):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities
        self.beta = min(1.0, self.beta + self.beta_increment)

    def __len__(self):
        return len(self.buffer)