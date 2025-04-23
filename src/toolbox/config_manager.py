# hybrid-llm-inference/src/toolbox/config_manager.py
import yaml
from pathlib import Path
from toolbox.logger import get_logger

class ConfigManager:
    def __init__(self, config_dir="configs"):
        """
        Initialize ConfigManager for loading YAML configurations.
        
        Args:
            config_dir (str): Directory containing YAML config files.
        """
        self.config_dir = Path(config_dir)
        self.logger = get_logger(__name__)
        self.configs = {}
    
    def load_config(self, config_file):
        """
        Load a YAML configuration file.
        
        Args:
            config_file (str): Name of the config file (e.g., 'model_config.yaml').
        
        Returns:
            dict: Parsed configuration.
        """
        config_path = self.config_dir / config_file
        if not config_path.exists():
            self.logger.error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.configs[config_file] = config
            self.logger.info(f"Loaded config: {config_path}")
            return config
        except yaml.YAMLError as e:
            self.logger.error(f"Failed to parse YAML: {e}")
            raise
    
    def get_config(self, config_file):
        """
        Get a previously loaded configuration.
        
        Args:
            config_file (str): Name of the config file.
        
        Returns:
            dict: Configuration or None if not loaded.
        """
        return self.configs.get(config_file)

