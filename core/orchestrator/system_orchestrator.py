#core/orchestrator/system_orchestrator.py

import asyncio
import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

# Constants for shutdown commands and confidence threshold
SHUTDOWN_COMMANDS = {"shutdown", "exit", "quit", "stop", "cancel", "terminate"}
INTENT_CONFIDENCE_THRESHOLD = 0.6

logger = logging.getLogger(__name__)

class ConfigValidationError(Exception):
    """Exception raised for errors in the configuration validation."""


class SystemOrchestrator:
    def __init__(self, config: dict):
        """
        Orchestrates subsystems including NLP, memory, voice, plugins, and authentication.

        Args:
            config (dict): Configuration dictionary with keys for subsystems.
        """
        self.config: dict = config
        self.nlp = None
        self.memory = None
        self.wake_word_detector = None
        self.voice_recognizer = None
        self.text_to_speech = None
        self.voice_interface = None
        self.plugin_orchestrator = None
        self.auth_manager = None
        self.running: bool = False
        self._running_lock = asyncio.Lock()

    async def validate_config(self) -> None:
        """
        Validates the configuration dictionary to ensure all required keys and types are present.

        Raises:
            ConfigValidationError: If required config keys are missing or invalid.
        """
        required_sections = [
            "nlp_config",
            "memory_config",
            "wake_word_config",
            "voice_recognizer_config",
            "text_to_speech_config",
        ]

        for section in required_sections:
            if section not in self.config:
                raise ConfigValidationError(f"Missing required config section: {section}")

            if not isinstance(self.config[section], dict):
                raise ConfigValidationError(f"Config section {section} must be a dict.")

        # Additional nested validations could be added here

    async def initialize_subsystems(self) -> None:
        """
        Initialize all subsystems: NLP, memory, voice, plugins, and authentication.
        """
        try:
            await self.validate_config()

            # Initialize NLP
            from core.intelligence import NLP
            self.nlp = NLP(self.config["nlp_config"])
            logger.info("NLP subsystem initialized")

            # Initialize memory
            from core.memory import Memory
            self.memory = Memory(self.config["memory_config"])
            logger.info("Memory subsystem initialized")

            # Initialize wake word detector
            from core.voice.wakeword import WakeWordDetector
            self.wake_word_detector = WakeWordDetector(self.config["wake_word_config"])
            logger.info("Wake word detector initialized")

            # Initialize voice recognizer
            from core.voice.speech_recognition import SpeechRecognizer
            self.voice_recognizer = SpeechRecognizer(self.config["voice_recognizer_config"])
            logger.info("Voice recognizer initialized")

            # Initialize text to speech
            from core.voice.text_to_speech import TextToSpeech
            self.text_to_speech = TextToSpeech(self.config["text_to_speech_config"])
            logger.info("Text to speech subsystem initialized")

            # Initialize voice interface if all voice components are present
            voice_components_present = all([
                self.wake_word_detector,
                self.voice_recognizer,
                self.text_to_speech,
            ])

            if voice_components_present:
                from core.voice.voice_interface import VoiceInterface
                self.voice_interface = VoiceInterface(
                    self.wake_word_detector,
                    self.voice_recognizer,
                    self.text_to_speech,
                )
                logger.info("Voice interface initialized")
            else:
                logger.warning("Voice components incomplete; voice interface not initialized")

            # Initialize plugin orchestrator (placeholder)
            # from core.plugins.plugin_orchestrator import PluginOrchestrator
            # self.plugin_orchestrator = PluginOrchestrator()

            # Initialize authentication manager (placeholder)
            # from core.auth import AuthManager
            # self.auth_manager = AuthManager()

        except ConfigValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize subsystems: {e}")
            raise

    async def shutdown(self) -> None:
        """
        Gracefully shutdown subsystems and release resources.
        """
        try:
            if self.voice_interface:
                await self._shutdown_voice_interface()
            if self.memory:
                await self.memory.close()
                logger.info("Memory subsystem closed")
            # Add plugin_orchestrator and auth_manager shutdown if implemented
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def _shutdown_voice_interface(self) -> None:
        """
        Shutdown voice interface and its components.
        """
        if self.voice_interface:
            await self.voice_interface.shutdown()
            logger.info("Voice interface shutdown complete")

    async def process_command(self, command: str) -> str:
        """
        Process a user command through NLP and plugins.

        Args:
            command (str): The input command string.

        Returns:
            str: The response text to the command.
        """
        logger.info(f"Processing command: {command}")

        if not command.strip():
            return "I did not receive any command."

        normalized_command = command.lower().strip()
        if normalized_command in SHUTDOWN_COMMANDS:
            await self._set_running(False)
            return "System is shutting down. Goodbye."

        # Process intent using NLP subsystem
        intent_data = await self._process_intent(command)
        if intent_data is None:
            return "I could not understand your command."

        intent, confidence = intent_data
        logger.debug(f"Intent recognized: {intent} with confidence {confidence}")

        if confidence < INTENT_CONFIDENCE_THRESHOLD:
            return "I am not confident enough to process that command."

        # Delegate intent handling to dedicated methods
        handler_name = f"_handle_intent_{intent}"
        handler = getattr(self, handler_name, None)
        if handler and callable(handler):
            try:
                response = await handler(command)
                return response
            except Exception as e:
                logger.error(f"Error handling intent '{intent}': {e}")
                return "An error occurred while processing your command."
        else:
            logger.warning(f"No handler for intent '{intent}'")
            return f"I'm sorry, I do not know how to handle the intent '{intent}'."

    async def _process_intent(self, command: str) -> Optional[tuple[str, float]]:
        """
        Calls the NLP subsystem to get intent and confidence.

        Args:
            command (str): The command to analyze.

        Returns:
            Optional[tuple[str, float]]: Tuple of (intent, confidence) or None if NLP not initialized.
        """
        if not self.nlp:
            logger.error("NLP subsystem not initialized")
            return None

        result = await self.nlp.process_command(command)
        if not result:
            return None

        # Expected result format: {"intent": str, "confidence": float}
        intent = result.get("intent")
        confidence = result.get("confidence", 0.0)
        if not intent:
            return None

        return intent, confidence

    async def _handle_intent_shutdown(self, command: str) -> str:
        await self._set_running(False)
        return "System is shutting down. Goodbye."

    async def _handle_intent_greeting(self, command: str) -> str:
        return "Greetings. How may I assist you today?"

    # Additional intent handlers should be implemented similarly

    async def _set_running(self, state: bool) -> None:
        """
        Sets the running state with concurrency lock.

        Args:
            state (bool): Desired running state.
        """
        async with self._running_lock:
            self.running = state

    async def run(self) -> None:
        """
        Main loop to start the orchestrator and listen for commands.
        """
        await self.initialize_subsystems()
        await self._set_running(True)

        logger.info("System Orchestrator is now running.")
        while True:
            async with self._running_lock:
                if not self.running:
                    break

            # Simplified example: Get input command from voice or CLI
            # Placeholder for actual input acquisition
            command = input("Command: ")
            response = await self.process_command(command)
            print(response)

        await self.shutdown()
        logger.info("System Orchestrator stopped.")
