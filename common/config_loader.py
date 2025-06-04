# marl/utils/config_loader.py
# This file is used to load the configuration file for the MARL training

import os
import yaml

def load_config(config_path):
    """
    Load configuration from a YAML file, including imported configurations.

    Args:
        config_path (str): Path to the main configuration file.

    Returns:
        dict: Merged configuration dictionary.
    """
    with open(config_path, 'r') as file:
        main_config = yaml.safe_load(file)
        base_path = os.path.dirname(config_path)
        imports = main_config.get('imports', [])
        for conf_file in imports:
            conf_path = os.path.join(base_path, conf_file)
            with open(conf_path, 'r') as f:
                conf = yaml.safe_load(f)
                main_config.update(conf)
    return main_config
