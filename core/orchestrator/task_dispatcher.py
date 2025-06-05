"""
Task dispatcher for Sebastian assistant.
"""
import logging
from typing import Dict, Any, Callable, Optional
from core.intelligence.intent_parser import IntentParser
from core.plugins import (
    greet, weather, music, alarm, reminder, session, device_control, memory
)
from core.task_manager import TaskManager
from core.memory.memory_interface import MemoryInterface

logger = logging.getLogger(__name__)

class PluginNotFoundError(Exception):
    """Raised when a requested plugin is not available."""
    pass

class PluginExecutionError(Exception):
    """Raised when a plugin fails to execute."""
    pass

class TaskDispatcher:
    """
    Responsible for routing parsed intents to the appropriate plugin handlers.
    Serves as the bridge between understanding user intent and executing actions.
    """
    
    def __init__(self, intent_parser: Optional[IntentParser] = None):
        self.intent_parser = intent_parser or IntentParser()
        self.plugins = {}
        logger.info("TaskDispatcher initialized")
        
    def register_plugin(self, intent: str, handler: Callable):
        """Register a plugin handler for a specific intent."""
        self.plugins[intent] = handler
        logger.info(f"Registered plugin for intent: {intent}")
        
    def dispatch(self, text: str) -> Dict[str, Any]:
        """
        Parse input and dispatch to appropriate plugin.
        
        Args:
            text (str): User input text
            
        Returns:
            Dict[str, Any]: Result from plugin execution
        """
        # Parse input to determine intent
        parsed = self.intent_parser.parse(text)
        intent = parsed["intent"]
        
        # Find handler for intent
        if intent not in self.plugins:
            logger.warning(f"No plugin found for intent: {intent}")
            raise PluginNotFoundError(f"No handler for intent '{intent}'")
        
        # Execute plugin
        try:
            handler = self.plugins[intent]
            result = handler(parsed)
            return result
        except Exception as e:
            logger.error(f"Plugin execution error: {e}")
            raise PluginExecutionError(f"Error executing plugin for '{intent}': {str(e)}")


# For backwards compatibility with the old import path
def dispatch_task(intent: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy function for dispatching tasks."""
    logger.warning("Using deprecated dispatch_task function. Use TaskDispatcher class instead.")
    # Create singleton instance for compatibility
    from core.orchestrator.plugin_orchestrator import get_plugin_orchestrator
    orchestrator = get_plugin_orchestrator()
    return orchestrator.execute_plugin(intent, data)
