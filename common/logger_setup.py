# marl/utils/logger_setup.py

import os
from datetime import datetime
import logging
import sys

def setup_logger(config):
    logger = logging.getLogger(config['project'])
    logger.setLevel(getattr(logging, config['logging']['log_level'].upper()))

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    if config['logging']['log_to_console']:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if config['logging']['log_to_file']:
        log_dir = config['logging']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{config['project']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
