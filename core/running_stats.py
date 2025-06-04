# marl_tsc/core/running_stats.py

# This file contains the RunningMeanStd class, which is used to compute running statistics
# such as mean and standard deviation for a given tensor. It is useful for normalizing observations
# in reinforcement learning environments. The class maintains a running mean, variance, and count
# of samples seen so far. The update method allows for updating these statistics with new data.

import torch

class RunningMeanStd:
    def __init__(self, epsilon=1e-5, shape=()):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape) * epsilon
        self.count = epsilon

    def update(self, x, var_min=1e-4):
        x = x.view(-1, *self.mean.shape)
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)  # Set unbiased=False
        batch_count = x.size(0)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        delta_sq = delta ** 2
        M2 = m_a + m_b + delta_sq * self.count * batch_count / total_count
        new_var = M2 / total_count

        # Prevent variance from being zero
        new_var = torch.clamp(new_var, min=var_min)

        self.mean = new_mean
        self.var = new_var
        self.count = total_count