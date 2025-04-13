import json
import os
from typing import Any, Dict, Optional

import yaml


class ConfigManager:
    """
    Class for managing configuration settings.
    This follows the Singleton pattern to ensure only one configuration instance exists.
    """

    _instance = None

    def __new__(cls, config_path: Optional[str] = None):
        """
        Create a new ConfigManager instance if one doesn't exist.

        Args:
            config_path: Path to the configuration file
        """
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the configuration file
        """
        # Only initialize once
        if self._initialized:
            return

        self.config_path = config_path
        self.config = {}

        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

        self._initialized = True

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Dictionary containing the configuration
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Determine file format from extension
        ext = os.path.splitext(config_path)[1].lower()

        if ext in [".yaml", ".yml"]:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        elif ext in [".json"]:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")

        self.config_path = config_path
        return self.config

    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        Save the current configuration to a file.

        Args:
            config_path: Path to save the configuration to (defaults to the loaded path)
        """
        save_path = config_path or self.config_path

        if not save_path:
            raise ValueError("No configuration path specified")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Determine file format from extension
        ext = os.path.splitext(save_path)[1].lower()

        if ext in [".yaml", ".yml"]:
            with open(save_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif ext in [".json"]:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key (can use dot notation for nested keys)
            default: Default value if the key doesn't exist

        Returns:
            Configuration value
        """
        # Support for nested keys using dot notation
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: Configuration key (can use dot notation for nested keys)
            value: Value to set
        """
        # Support for nested keys using dot notation
        keys = key.split(".")
        config = self.config

        # Navigate to the correct level
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

    def get_all(self) -> Dict[str, Any]:
        """
        Get the entire configuration.

        Returns:
            Dictionary containing the entire configuration
        """
        return self.config.copy()


# Default configuration
DEFAULT_CONFIG = {
    "data": {
        "data_dir": "data",
        "data_file": "example_data.csv",
        # Flexible column mappings
        "columns": {
            # Input parameters (features)
            "parameters": {
                "x1": "blads",
                "x2": "baojiao",
                "x3": "angle_in",
                "x4": "angle_out",
            },
            # Target variables (objectives)
            "targets": {
                "y1": {
                    "column": "head",
                    "type": "maximize",  # We want to maximize head/power
                },
                "y2": {
                    "column": "efficiency",
                    "type": "maximize",  # We want to maximize efficiency
                },
            },
            # Specify which parameters should be treated as integers
            "integer_parameters": ["x1"],
        },
    },
    "models": {
        "y1_model": {
            "type": "random_forest",
            "params": {"n_estimators": 100, "random_state": 42},
        },
        "y2_model": {
            "type": "random_forest",
            "params": {"n_estimators": 100, "random_state": 42},
        },
    },
    "optimization": {
        "algorithm": "nsga2",
        "algorithm_params": {
            "pop_size": 200,
            "n_offsprings": 50,
            "n_gen": 200,
            "seed": 1,
        },
        # This will be determined automatically from integer_parameters in data.columns
        "integer_vars": [],
    },
    "visualization": {
        "dpi": 300,
        "show_plots": True,
        "save_plots": True,
        "output_dir": "results",
    },
}


def init_config(config_path: str) -> ConfigManager:
    """
    Initialize the configuration with default values.

    Args:
        config_path: Path to save the default configuration

    Returns:
        ConfigManager instance
    """
    config = ConfigManager()
    config.config = DEFAULT_CONFIG.copy()
    config.config_path = config_path
    config.save_config()
    return config
