import os

from trainer import start_training_run, logger
from util import load_config


config_path = "config.yaml"

# Load the config
config = load_config(config_path)

# Create the logging directory
os.makedirs(config["training"]["log_dir"], exist_ok=True)

# Change the logging level
logger.setLevel(config["logging_level"])

# Start a training run
start_training_run(config)
