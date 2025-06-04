# marl_tsc/utils/monitoring_params.py
# This file contains a function that initializes the monitoring tool for the training process.

def init_monitoring(config):
    if config.get('use_wandb', False):
        import wandb
        wandb.init(project=config['project'])
        wandb.config.update(config)
        monitor = wandb
    elif config.get('use_neptune', False):
        import neptune.new as neptune
        run = neptune.init(project=config['project'])
        run['config'] = config
        monitor = run
    else:
        monitor = None
    return monitor