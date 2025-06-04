# scripts/run_training.py

from marl_tsc.utils.config_loader import load_config
from marl_tsc.scripts.trainer import Trainer

def main():
    config_path = 'configs/main_config.yaml'
    config = load_config(config_path)
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
