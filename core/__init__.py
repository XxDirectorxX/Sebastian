# core/__init__.py

import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """
    Load and parse the assistant's YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML config file.
        
    Returns:
        dict: Parsed configuration dictionary.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    try:
        with config_file.open("r") as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
    except yaml.YAMLError as ye:
        logger.error(f"YAML parsing error in {config_path}: {ye}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading config {config_path}: {e}")
        raise

    # Basic validation or default fills could be done here
    # e.g. config.setdefault('access_control', {})
    
    return config


