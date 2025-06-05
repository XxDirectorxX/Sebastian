"""
Main entry point for Sebastian AI assistant.

Handles initialization, configuration loading, and execution modes.
"""
import logging
import logging.config
import signal
import sys
import asyncio
import yaml 
from typing import Optional, Dict, Any
from pathlib import Path
from core.plugin_utils import load_plugins

load_plugins("core/plugins")  # Adjust path if plugins are under core/plugins
logger.info("Plugins loaded successfully.")


# Import the actual SebastianSystem orchestrator
from core.orchestrator.system_orchestrator import SebastianSystem
from core.config.config_loader import ConfigLoader

# Configure basic logging
# Moved logging setup to a more central place or ensure it's configured before use.
# For now, let's ensure it's set up here if not elsewhere.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
        # logging.FileHandler("sebastian_runtime.log") # Optional: for file logging
    ]
)
logger = logging.getLogger("Sebastian.Run")

def persona_say(message: str, error: bool = False):
    # This will eventually be enhanced by the personality and voice synthesis modules
    prefix = "Sebastian: "
    # Ensure logger is available, might not be fully configured if persona_say is called very early
    current_logger = logging.getLogger("Sebastian.Persona") if logger.handlers else logging.getLogger("Sebastian")

    if error:
        # Consider logging errors more formally via the logger
        print(f"{prefix}My sincerest apologies, but an error has occurred: {message}")
        current_logger.error(message)
    else:
        print(f"{prefix}{message}")
        current_logger.info(f"Said: {message}")

class GracefulExit:
    """Handle graceful system shutdown on signals."""
    def __init__(self):
        self.exit_pending = False
        # Signal handlers will be set in main_async after system is created,
        # to allow them to interact with the system object.
        # self.system_to_notify: Optional[SebastianSystem] = None # This was an idea, but direct signal handling in main_async is cleaner

    def initiate_exit_sync(self, signum, frame): # Renamed to avoid conflict if used elsewhere
        # This synchronous version is a fallback or for non-async contexts.
        # The primary signal handling will be in main_async.
        if not self.exit_pending:
            self.exit_pending = True
            persona_say("It appears my services are no longer required at this moment. Shutting down with precision, My Lord.")
            logger.info(f"Synchronous shutdown initiated by signal {signum}.")
            # if self.system_to_notify:
            #    self.system_to_notify.running = False # This is problematic from a signal handler directly into asyncio
            # For a truly synchronous exit from a signal handler (if not in asyncio loop):
            # sys.exit(0) 
            # However, we aim for graceful asyncio shutdown.

# It would be prudent to move configuration and logging setup to their respective modules
# e.g., core.config.loader.py and core.config.logging_setup.py

def setup_logging(config_path: Optional[str] = "core/config/logging.yaml") -> logging.Logger:
    """
    Setup logging configuration.
    """
    log_conf_path = Path(config_path) if config_path else None
    if log_conf_path and log_conf_path.exists():
        try:
            with open(log_conf_path, "r") as f:
                config_dict = yaml.safe_load(f)
            logging.config.dictConfig(config_dict)
            # Return the specifically configured logger
            return logging.getLogger(config_dict.get("root", {}).get("handlers", ["Sebastian"])[0] if "root" in config_dict else "Sebastian")
        except Exception as e:
            persona_say(f"Failed to load logging configuration from {log_conf_path}: {e}. Proceeding with basic settings.", error=True)
            # Fallback to basic config if file load fails
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            return logging.getLogger("Sebastian.Fallback")
    else:
        if config_path: # Only warn if a specific path was given and not found
            persona_say(f"Logging configuration file '{config_path}' not found. Proceeding with default logging settings.", error=True)
        # Basic config if no file or path is found
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        return logging.getLogger("Sebastian.Default")


DEFAULT_CONFIG_PATH = "core/config/assistant_config.yaml"
DEFAULT_PERSONA_PROFILE_PATH = "assets/personas/persona_profile.yaml"

def create_default_config_structure() -> Dict[str, Any]:
    """
    Defines the structure for a default configuration.
    This should ideally be loaded from a template or a dedicated defaults file.
    """
    return {
        "assistant_name": "Sebastian",
        "mode": "interactive", # Default mode
        "voice": {
            "enabled": False, # Default to False until fully implemented
            "wake_word": "sebastian",
            "voice_model": "assets/voice_models/sebastian_voice_model.pth",
            "sample_rate": 16000,
            "continuous_listening": False
        },
        "vision": {
            "enabled": False,
            "face_recognition": False,
            "object_detection": False
        },
        "memory": {
            "short_term_capacity": 100,
            "long_term_storage": "sqlite", # e.g., core/memory/data/sebastian_memory.db
            "embeddings_model": "sentence-transformers/all-mpnet-base-v2"
        },
        "security": {
            "access_control": False,
            "encryption": False, # For logs and sensitive data
            "logging_level": "INFO" # This should be driven by logging.yaml
        },
        "personality": {
            "base_formality": 8,
            "simulation_fidelity": "high",
            "persona_file": DEFAULT_PERSONA_PROFILE_PATH
        },
        "paths": { # Centralize important paths
            "assets": "assets",
            "logs": "logs",
            "core_config": "core/config"
        }
    }

def save_config(config: Dict[str, Any], config_path: str = DEFAULT_CONFIG_PATH) -> None:
    """Saves the configuration to a YAML file."""
    try:
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(config, f, indent=4, sort_keys=False)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        persona_say(f"Failed to save configuration to {config_path}: {e}", error=True)


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load configuration from YAML file with validation and default creation.
    """
    path = Path(config_path)
    if not path.exists():
        persona_say(f"Configuration file '{config_path}' not found. Creating default configuration.", error=False) # Not an error, but an action
        logger.info(f"Creating default configuration at {config_path}")
        config = create_default_config_structure()
        save_config(config, config_path)
    else:
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)
            if config is None: # File might be empty
                persona_say(f"Configuration file '{config_path}' is empty. Recreating default configuration.", error=True)
                config = create_default_config_structure()
                save_config(config, config_path)
        except yaml.YAMLError as e:
            persona_say(f"Error parsing configuration file '{config_path}': {e}. Attempting to recreate.", error=True)
            config = create_default_config_structure()
            save_config(config, config_path) # Overwrite corrupted file
        except Exception as e: # Catch other file reading errors
            persona_say(f"Failed to load configuration from '{config_path}': {e}. Using in-memory default.", error=True)
            config = create_default_config_structure() # Use in-memory, don't save over potentially good file without explicit instruction

    validate_configuration(config, config_path) # Pass path for context in validation
    return config

def validate_configuration(config: Dict[str, Any], config_path_for_context: str) -> None:
    """
    Validate essential configuration settings.
    """
    default_struct = create_default_config_structure()
    essential_sections = ["voice", "memory", "personality", "paths"]
    updated = False

    for section in essential_sections:
        if section not in config:
            persona_say(f"Configuration missing essential section: '{section}' in '{config_path_for_context}'. Using defaults for this section.", error=True)
            config[section] = default_struct[section]
            updated = True

    # Validate persona file existence
    persona_file_path_str = config.get("personality", {}).get("persona_file", DEFAULT_PERSONA_PROFILE_PATH)
    persona_file = Path(persona_file_path_str)
    if not persona_file.exists():
        persona_say(f"Persona profile '{persona_file_path_str}' not found. This is essential for my defined character. Please ensure the file exists or a default can be created.", error=True)
        # Potentially create a default persona_profile.yaml if feasible, or halt.
        # For now, we'll assume it's a critical error if not found and not auto-created.
        # config["personality"]["persona_file"] = None # Or some indicator it's missing

    # Example: Validate voice model path if voice is enabled
    if config.get("voice", {}).get("enabled", False):
        voice_model_path_str = config.get("voice", {}).get("voice_model")
        if not voice_model_path_str or not Path(voice_model_path_str).exists():
            persona_say(f"Voice model path '{voice_model_path_str}' not found, yet voice is enabled. Voice capabilities will be impaired.", error=True)
            # config["voice"]["enabled"] = False # Disable voice if model is missing

    if updated:
        persona_say(f"Configuration '{config_path_for_context}' was updated with default values for missing sections. It is advisable to review it.", error=False)
        save_config(config, config_path_for_context)


async def run_interactive_mode(config: Dict[str, Any]):
    """Runs Sebastian in interactive command-line mode."""
    logger.info("Starting Sebastian in Interactive Mode...")
    sebastian_system = SebastianSystem(config)
    await sebastian_system.initialize_subsystems()

    try:
        while sebastian_system.running:
            command_text = await asyncio.to_thread(input, "My Lord> ")
            if command_text.lower().strip() in ["exit", "quit", "shutdown"]:
                logger.info("Shutdown command received. Exiting interactive mode.")
                sebastian_system.running = False # Signal system to prepare for shutdown
                break 
            
            if command_text:
                response = await sebastian_system.process_command(command_text, user_id="cli_user")
                if response: # Only print if there's a response string
                    print(f"Sebastian: {response}")
            await asyncio.sleep(0.01) # Yield control briefly
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
        sebastian_system.running = False
    except Exception as e:
        logger.error(f"An error occurred in interactive mode: {e}", exc_info=True)
        sebastian_system.running = False
    finally:
        logger.info("Shutting down Sebastian system from interactive mode...")
        await sebastian_system.shutdown()
        logger.info("Interactive mode finished.")

async def run_voice_mode(config: Dict[str, Any]):
    """Runs Sebastian in voice interaction mode."""
    logger.info("Starting Sebastian in Voice Mode...")
    sebastian_system = SebastianSystem(config)
    
    try:
        await sebastian_system.initialize_subsystems()

        if sebastian_system.voice_interface:
            logger.info("Voice interface initialized. Attempting to start voice interaction.")
            await sebastian_system.start_voice_interaction()
            # The voice interaction loop runs in the background via tasks managed by VoiceInterface.
            # The main loop here just keeps the application alive.
            while sebastian_system.running:
                await asyncio.sleep(0.5) # Keep alive, check running status periodically
        else:
            logger.error("VoiceInterface failed to initialize. Cannot start voice mode. Please check configuration and logs.")
            # Optionally, fall back to interactive mode or exit
            logger.info("Exiting due to VoiceInterface initialization failure.")
            sebastian_system.running = False # Ensure shutdown sequence is called

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down voice mode...")
        sebastian_system.running = False
    except Exception as e:
        logger.error(f"An critical error occurred in voice mode setup or main loop: {e}", exc_info=True)
        sebastian_system.running = False # Ensure shutdown on critical error
    finally:
        logger.info("Shutting down Sebastian system from voice mode...")
        await sebastian_system.shutdown() # This will also stop the voice_interface loop
        logger.info("Voice mode finished.")


async def main():
    """Main entry point for Sebastian AI."""
    config_loader = ConfigLoader()
    config = config_loader.get_config()

    # Update logging level from config if specified
    log_level_str = config.get("system", {}).get("logging_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Reconfigure root logger level and potentially other handlers if needed
    # For simplicity, just setting the level of our main logger and the root.
    logging.getLogger().setLevel(log_level) # Set root logger level
    for handler in logging.getLogger().handlers: # Apply to existing handlers
        handler.setLevel(log_level)
    logger.info(f"Logging level set to {log_level_str}.")


    mode = config.get("system", {}).get("mode", "interactive").lower()
    logger.info(f"Sebastian starting in '{mode}' mode.")

    if mode == "interactive":
        await run_interactive_mode(config)
    elif mode == "voice":
        await run_voice_mode(config)
    else:
        logger.error(f"Unknown mode: {mode}. Defaulting to interactive.")
        await run_interactive_mode(config)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Unhandled exception in main asyncio run: {e}", exc_info=True)
        sys.exit(1)
