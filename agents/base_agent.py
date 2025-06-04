# marl/agents/base_agent.py

from abc import ABC, abstractmethod
import torch.nn as nn
from marl_tsc.utils.other_utils import RunningMeanStd

class BaseAgent(nn.Module, ABC):
    def __init__(self, config, agent_id, obs_dim, continuous_dims):
        super(BaseAgent, self).__init__()
        self.config = config
        self.agent_id = agent_id
        self.continuous_dims = continuous_dims  # Indices of continuous observation components

        if config['advanced_techniques'].get('normalize_observations', False):
            self.obs_rms = RunningMeanStd(shape=(len(self.continuous_dims),))
        else:
            self.obs_rms = None



    @abstractmethod
    def select_action(self, observation):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass
