"""
Helper utility for loading dataset-specific JSON configurations 
and dynamically applying them to the global settings object.
"""

import json
from pathlib import Path
from config.settings import settings
from config.logging_config import logger

def load_and_apply_config(dataset_name: str, config_type: str) -> dict:
    """
    Load dataset-specific JSON configuration from the config/ directory
    and apply the values to the global settings object.
    
    Args:
        dataset_name: 'bioasq' or 'medaesqa'
        config_type: 'retrieval' or 'generation'
        
    Returns:
        dict: The loaded configuration dictionary (empty if file does not exist)
    """
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    config_filename = f"{dataset_name.lower()}_{config_type.lower()}.json"
    config_path = project_root / "config" / config_filename
    
    if not config_path.exists():
        logger.info(f"No specific config file found at {config_path}. Using settings defaults.")
        return {}
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            
        logger.info(f"Loaded config from {config_path}: {config_data}")
        
        # Apply to global settings object
        for key, val in config_data.items():
            settings_key = key.upper()
            if hasattr(settings, settings_key):
                old_val = getattr(settings, settings_key)
                setattr(settings, settings_key, val)
                logger.debug(f"Overrode settings.{settings_key}: {old_val} -> {val}")
            else:
                logger.warning(f"Settings object has no attribute '{settings_key}'. Ignored.")
                
        return config_data
    except Exception as e:
        logger.error(f"Error loading or applying config from {config_path}: {e}")
        return {}
