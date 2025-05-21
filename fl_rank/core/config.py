# fl_rank/core/config.py
"""
Configuration management for fl-rank.
"""

from typing import Any, Dict, Optional


class Config:
    """
    Configuration manager for fl-rank components.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration with optional dictionary.
        
        Args:
            config_dict: Initial configuration dictionary
        """
        self._config = config_dict or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Any: Configuration value
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with a dictionary.
        
        Args:
            config_dict: Dictionary with configuration updates
        """
        self._config.update(config_dict)
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Get the configuration as a dictionary.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return self._config.copy()