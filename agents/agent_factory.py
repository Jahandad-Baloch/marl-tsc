# marl/agents/agent_factory.py
# This file contains the agent factory function to create agents based on the configuration file.

from marl_tsc.agents.maddpg import MADDPGAgent
from marl_tsc.agents.mappo_agent import MAPPOAgent
# Import other agent classes as needed



def create_agent(config, agent_id, obs_space, action_space):
    algorithm = config['model']['algorithm']
    if algorithm == 'MAPPO':
        return MAPPOAgent(config, agent_id, obs_space, action_space)
    elif algorithm == 'MADDPG':
        return MADDPGAgent(config, agent_id, obs_space, action_space)
    elif algorithm == 'QMix':
        pass
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
