core/config/config_loader.py

import yaml
import os
import logging
from typing import Dict, Any, Optional
from threading import Lock

logger = logging.getLogger("Sebastian.ConfigLoader")

class ConfigLoaderError(Exception):
    """Custom exception for ConfigLoader critical errors."""
    pass

class ConfigLoader:
    """
    Singleton class to load and provide access to configuration settings for Sebastian.
    Supports default config creation, flexible file location, and thread-safe loading.
    """

    _instance: Optional['ConfigLoader'] = None
    _lock: Lock = Lock()

    def __new__(cls, config_path: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigLoader, cls).__new__(cls)
                cls._instance._config_path = config_path or cls._determine_default_path()
                cls._instance._config = None
                cls._instance._load_config()
            else:
                # If called with a different config_path, reload config accordingly
                if config_path and config_path != cls._instance._config_path:
                    cls._instance._config_path = config_path
                    cls._instance._load_config()
            return cls._instance

    @staticmethod
    def _determine_default_path() -> str:
        # Compute project root by moving three levels up from this file location
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_path = os.path.join(project_root, "core", "config", "assistant_config.yaml")
        return default_path

    def _load_config(self) -> None:
        """
        Loads configuration from YAML file, creating default config if necessary.
        """
        path = self._config_path
        logger.info(f"Loading configuration from: {path}")

        if not os.path.exists(path):
            logger.warning(f"Config file not found at {path}. Attempting to create default config.")
            try:
                self._create_default_config(path)
            except Exception as e:
                logger.error(f"Failed to create default config at {path}: {e}", exc_info=True)
                # Optionally, raise or fallback to empty config
                self._config = {}
                return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    logger.warning(f"Config file {path} is empty or malformed. Using empty config.")
                    self._config = {}
                else:
                    self._config = data
                    logger.info("Configuration loaded successfully.")
        except (yaml.YAMLError, IOError) as e:
            logger.error(f"Error loading config file {path}: {e}", exc_info=True)
            self._config = {}

    def _create_default_config(self, path: str) -> None:
        """
        Creates a default configuration file at specified path if it does not exist.
        """
        if os.path.exists(path):
            logger.debug("Default config creation skipped; file already exists.")
            return
        default_config = {
            "system": {
                "name": "Sebastian",
                "version": "0.1.0",
                "mode": "interactive",
                "logging_level": "INFO"
            },
            "intelligence": {
                "nlp_engine": {
                    "provider": "spacy",
                    "spacy_model_name": "en_core_web_sm",
                    "intents": {
                        "greet": ["hello", "hi", "greetings", "hey", "good morning", "good afternoon", "good evening"],
                        "get_time": ["time", "what time is it", "current time"],
                        "get_weather": ["weather", "forecast", "what's the weather like"],
                        "exit": ["exit", "quit", "goodbye", "bye", "shutdown"],
                        "set_user_name": ["call me", "my name is"],
                        "create_memory": ["remember that", "make a note", "don't forget"],
                        "recall_memories": ["what do you remember", "recall my information", "tell me what you know"],
                        "control_device": ["turn on", "turn off", "activate", "deactivate", "set"],
                        "shutdown_system": ["shutdown system", "power down", "initiate full shutdown"]
                    }
                }
            },
            "memory": {
                "short_term": {
                    "provider": "in_memory",
                    "max_history_per_session": 20
                },
                "long_term": {
                    "provider": "sqlite",
                    "sqlite_db_path": "sebastian_memory.db"
                }
            },
            "voice": {
                "wake_word_engine": "porcupine",
                "porcupine": {
                    "access_key": "YOUR_PICOVOICE_ACCESS_KEY_HERE",
                    "keyword_paths": ["assets/wakewords/sebastian_windows.ppn"],
                    "sensitivity": 0.6,
                    "library_path": None,
                    "model_path": None
                },
                "stt_engine": "whisper",
                "whisper": {
                    "model_size": "tiny.en",
                    "language": "en",
                    "device": None
                },
                "tts_engine": "coqui_tts",
                "coqui_tts": {
                    "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
                    "vocoder_name": None,
                    "device": None
                },
                "voice_interface": {
                    "input_device_index": None,
                    "output_device_index": None,
                    "sample_rate": 16000,
                    "channels": 1,
                    "command_recording_duration_seconds": 5.0,
                    "ack_sound_path": "assets/sounds/ack.wav",
                    "error_sound_path": "assets/sounds/error.wav"
                }
            }
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, sort_keys=False, indent=2)
        logger.info(f"Default configuration created at {path}")

    @property
    def config(self) -> Dict[str, Any]:
        """
        Returns the loaded configuration dictionary.
        """
        if self._config is None:
            logger.warning("Configuration accessed before loading; loading now.")
            self._load_config()
        return self._config or {}

    def reload(self) -> None:
        """
        Forces a reload of the configuration file from disk.
        """
        with self._lock:
            self._load_config()

    def get_setting(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Retrieve a specific setting or whole section from the configuration.
        """
        cfg = self.config
        section_data = cfg.get(section)
        if section_data is None:
            return default
        if key is None:
            return section_data
        return section_data.get(key, default)

# Optional test block with safe logging config
if __name__ == "__main__":
    import sys
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stdout
        )

    loader = ConfigLoader()
    print(f"Config loaded: {loader.get_setting('system', 'name', 'N/A')}")
    print(f"Singleton check: {loader is ConfigLoader()}")
    print(f"All 'system' settings: {loader.get_setting('system')}")
    print(f"Non-existent section fallback: {loader.get_setting('nonexistent', 'key', 'fallback')}")
    print(f"Config file path: {loader._config_path}")
