import os
import yaml
import logging
from threading import RLock
from typing import Optional, Dict, Any

from pydantic import BaseModel, ValidationError, Field

logger = logging.getLogger("Sebastian.ConfigLoader")


class ConfigLoaderError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


class SystemConfig(BaseModel):
    name: str = "Sebastian"
    version: str = "0.1.0"
    mode: str = "interactive"
    logging_level: str = "INFO"


class NLPConfig(BaseModel):
    provider: str = "spacy"
    spacy_model_name: Optional[str] = None
    intents: Dict[str, list] = Field(default_factory=dict)


class IntelligenceConfig(BaseModel):
    nlp_engine: NLPConfig = NLPConfig()


class ShortTermMemoryConfig(BaseModel):
    ttl_seconds: int = 300


class ConfigLoader:
    _lock = RLock()

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._determine_default_path()
        self.config_data = None

    def _determine_default_path(self) -> str:
        """Determine the default configuration path."""
        default_path = os.getenv("SEBASTIAN_CONFIG_PATH", "./core/config/assistant_config.yaml")
        if not os.path.exists(default_path):
            raise ConfigLoaderError(f"Default configuration file not found at {default_path}")
        return default_path

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from the YAML file."""
        with self._lock:
            if self.config_data:
                return self.config_data

            try:
                with open(self.config_path, "r") as file:
                    raw_config = yaml.safe_load(file)
                    self.config_data = self._validate_config(raw_config)
                    logger.info("Configuration loaded successfully.")
                    return self.config_data
            except FileNotFoundError:
                raise ConfigLoaderError(f"Configuration file not found at {self.config_path}")
            except yaml.YAMLError as e:
                raise ConfigLoaderError(f"Error parsing YAML configuration: {e}")

    def _validate_config(self, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the configuration using Pydantic models."""
        try:
            system_config = SystemConfig(**raw_config.get("system", {}))
            intelligence_config = IntelligenceConfig(**raw_config.get("intelligence", {}))
            memory_config = ShortTermMemoryConfig(**raw_config.get("memory", {}))

            return {
                "system": system_config.dict(),
                "intelligence": intelligence_config.dict(),
                "memory": memory_config.dict(),
            }
        except ValidationError as e:
            raise ConfigLoaderError(f"Configuration validation failed: {e}")

    def reload_config(self) -> None:
        """Reload the configuration."""
        with self._lock:
            self.config_data = None
            self.load_config()


if __name__ == "__main__":
    import sys
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stdout
        )

    try:
        loader = ConfigLoader()
        print(f"System name: {loader.get_setting('system', 'name', 'N/A')}")
        print(f"Mode: {loader.get_setting('system', 'mode')}")
    except ConfigLoaderError as e:
        print(f"Config loading failed: {e}")
