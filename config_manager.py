import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

@dataclass
class RoutingConfig:
    category: str
    keywords: List[str]
    model_name: str
    prompt_template: str

class ConfigManager:
    def __init__(self, config_file: str = "routing_config.json"):
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = {
                    "categories": {
                        "finance": {
                            "keywords": ["money", "invest", "stock", "market"],
                            "model": "claude",
                            "prompt_template": "You are a financial expert..."
                        },
                        "literature": {
                            "keywords": ["book", "novel", "poem", "author"],
                            "model": "chatgpt",
                            "prompt_template": "You are a literature expert..."
                        },
                        "general": {
                            "keywords": [],
                            "model": "local",
                            "prompt_template": "Please analyze the following..."
                        }
                    },
                    "admin_password": "admin123"  # Change this in production
                }
                self._save_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise

    def _save_config(self) -> None:
        """Save configuration to JSON file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving config: {str(e)}")
            raise

    def add_category(self, category: str, keywords: List[str], model: str, prompt_template: str) -> None:
        """Add or update a category configuration."""
        self.config["categories"][category] = {
            "keywords": keywords,
            "model": model,
            "prompt_template": prompt_template
        }
        self._save_config()

    def remove_category(self, category: str) -> None:
        """Remove a category configuration."""
        if category in self.config["categories"]:
            del self.config["categories"][category]
            self._save_config()

    def get_categories(self) -> Dict:
        """Get all category configurations."""
        return self.config["categories"]

    def verify_admin_password(self, password: str) -> bool:
        """Verify admin password."""
        return password == self.config["admin_password"]
def get_categories(self) -> Dict:
    """
    Get all category configurations.
    
    Returns:
        Dict: Dictionary containing category configurations
        
    Raises:
        RuntimeError: If configuration cannot be loaded
    """
    try:
        if not hasattr(self, 'config'):
            self._load_config()
        return self.config.get("categories", {})
    except Exception as e:
        self.logger.error(f"Error retrieving categories: {str(e)}")
        return {
            "general": {
                "keywords": [],
                "model": "local",
                "prompt_template": "Please analyze the following question: {input}"
            }
        }
