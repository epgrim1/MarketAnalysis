#!/usr/bin/env python3
# src/utils/config.py

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

class ConfigManager:
    """Manage configuration settings for market analysis."""
    
    DEFAULT_CONFIG = {
        "paths": {
            "data_dir": "data",
            "raw_data": "data/raw",
            "processed_data": "data/processed",
            "output": "data/output",
            "logs": "logs"
        },
        "api": {
            "polygon_key": "",
            "tradingview_screener": "america",
            "default_exchange": "AMEX"
        },
        "analysis": {
            "lookback_days": 252,
            "min_data_points": 126,
            "volatility_window": 20,
            "correlation_window": 63
        },
        "sector_rotation": {
            "min_volume": 1000,
            "min_relative_strength": 0.7,
            "momentum_threshold": 0.5,
            "technical_weights": {
                "trend": 0.4,
                "momentum": 0.3,
                "volatility": 0.3
            }
        },
        "model": {
            "train_test_split": 0.2,
            "random_state": 42,
            "rf_params": {
                "n_estimators": 200,
                "max_depth": 12,
                "min_samples_split": 5,
                "min_samples_leaf": 2
            }
        },
        "sectors": {
            "XLK": "Technology",
            "XLF": "Financials",
            "XLE": "Energy",
            "XLV": "Healthcare",
            "XLP": "Consumer Staples",
            "XLI": "Industrials",
            "XLB": "Materials",
            "XLU": "Utilities",
            "XLY": "Consumer Discretionary",
            "SMH": "Semiconductors",
            "XRT": "Retail"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_rotation": "1 week"
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ConfigManager with optional custom config path.
        
        Args:
            config_path (str, optional): Path to custom config file
        """
        self.config_path = config_path or "config/config.json"
        self.config = self.DEFAULT_CONFIG.copy()
        self._setup_logging()
        self.load_config()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_dir = Path(self.config["paths"]["logs"])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=self.config["logging"]["level"],
            format=self.config["logging"]["format"],
            handlers=[
                logging.FileHandler(
                    log_dir / f"config_{datetime.now().strftime('%Y%m%d')}.log"
                ),
                logging.StreamHandler()
            ]
        )

    def load_config(self) -> None:
        """Load configuration from file if it exists."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                self._update_nested_dict(self.config, user_config)
                logging.info(f"Loaded configuration from {self.config_path}")
            else:
                self.save_config()
                logging.info(f"Created new configuration file at {self.config_path}")
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")

    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            logging.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key (str): Configuration key (can be nested using dots)
            default (Any): Default value if key not found
            
        Returns:
            Any: Configuration value or default
        """
        try:
            value = self.config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key (str): Configuration key (can be nested using dots)
            value (Any): Value to set
        """
        try:
            keys = key.split('.')
            d = self.config
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value
            self.save_config()
            logging.info(f"Updated configuration: {key} = {value}")
        except Exception as e:
            logging.error(f"Error setting configuration {key}: {e}")

    @staticmethod
    def _update_nested_dict(d: Dict, u: Dict) -> Dict:
        """Recursively update nested dictionary."""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = ConfigManager._update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def validate_paths(self) -> None:
        """Validate and create necessary directories."""
        try:
            for path_key, path_value in self.config["paths"].items():
                path = Path(path_value)
                path.mkdir(parents=True, exist_ok=True)
                logging.info(f"Validated path: {path_key} = {path}")
        except Exception as e:
            logging.error(f"Error validating paths: {e}")

    def get_sector_config(self, symbol: str) -> Dict:
        """
        Get sector-specific configuration.
        
        Args:
            symbol (str): Sector symbol
            
        Returns:
            Dict: Sector configuration
        """
        base_config = self.config["sector_rotation"].copy()
        
        # Add sector-specific adjustments if needed
        if symbol == "XLU":  # Example: Different thresholds for utilities
            base_config["min_relative_strength"] = 0.6
        elif symbol == "SMH":  # Example: Different volume requirements for semis
            base_config["min_volume"] = 2000
            
        return base_config

# Create global config instance
config = ConfigManager()

def load_config(config_path: str = None) -> ConfigManager:
    """Load configuration from specified path."""
    return ConfigManager(config_path)

def save_config() -> None:
    """Save current configuration."""
    config.save_config()

def get_default_config() -> Dict:
    """Get default configuration dictionary."""
    return ConfigManager.DEFAULT_CONFIG.copy()