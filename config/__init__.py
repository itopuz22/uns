"""Configuration management for Industrial Quality Control System."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# Default configuration file path
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


class ConfigManager:
    """Manages system configuration with validation and defaults."""

    _instance: Optional['ConfigManager'] = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: Optional[Path] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config(config_path or DEFAULT_CONFIG_PATH)
        return cls._instance

    def _load_config(self, config_path: Path) -> None:
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Example: config.get('camera.resolution.capture_width')
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self._config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    @property
    def camera(self) -> Dict[str, Any]:
        return self._config.get('camera', {})

    @property
    def gpio(self) -> Dict[str, Any]:
        return self._config.get('gpio', {})

    @property
    def model(self) -> Dict[str, Any]:
        return self._config.get('model', {})

    @property
    def ftp(self) -> Dict[str, Any]:
        return self._config.get('ftp', {})

    @property
    def dashboard(self) -> Dict[str, Any]:
        return self._config.get('dashboard', {})

    @property
    def training(self) -> Dict[str, Any]:
        return self._config.get('training', {})

    def reload(self, config_path: Optional[Path] = None) -> None:
        """Reload configuration from file."""
        self._load_config(config_path or DEFAULT_CONFIG_PATH)


def get_config(config_path: Optional[Path] = None) -> ConfigManager:
    """Get the singleton configuration manager instance."""
    return ConfigManager(config_path)
