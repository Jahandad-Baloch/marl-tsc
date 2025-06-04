# marl/algorithms/algorithm_factory.py
# This file contains a factory function that creates an instance of an algorithm based on the provided algorithm name.

def create_algorithm(config, env, logger=None):
    algorithm_name = config['model']['algorithm']
    if algorithm_name == 'MAPPO':
        from marl_tsc.algorithms.mappo import MAPPO
        return MAPPO(config, env, logger)
    elif algorithm_name == 'MADDPG':
        from marl_tsc.algorithms.maddpg import MADDPG
        return MADDPG(config, env, logger)
    elif algorithm_name == 'QMix':
        from marl_tsc.algorithms.qmix import QMix
        return QMix(config, env, logger)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")
