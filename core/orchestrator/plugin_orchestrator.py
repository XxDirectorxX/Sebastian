"""
Plugin orchestrator for Sebastian assistant.
"""

import logging
import importlib.util
import os
from typing import Dict, Any, Optional
from core.memory.short_term import ShortTermMemory
from core.memory.long_term import LongTermMemory
from core.task_dispatcher import dispatch_task, PluginExecutionError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Singleton instance
_plugin_orchestrator = None


def get_plugin_orchestrator():
    """Get singleton instance of PluginOrchestrator."""
    global _plugin_orchestrator
    if _plugin_orchestrator is None:
        _plugin_orchestrator = PluginOrchestrator()
    return _plugin_orchestrator


class PluginOrchestrator:
    """
    Manages plugin discovery, loading, and execution.
    Maintains a registry of available plugins and their capabilities.
    """

    def __init__(self, plugin_dir: str = "core/plugins"):
        self.plugin_dir = plugin_dir
        self.plugins = {}
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory()
        self._discover_plugins()
        logger.info(f"PluginOrchestrator initialized with {len(self.plugins)} plugins")

    def _discover_plugins(self):
        """Discover and load available plugins."""
        if not os.path.exists(self.plugin_dir):
            logger.warning(f"Plugin directory not found: {self.plugin_dir}")
            return

        for filename in os.listdir(self.plugin_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                plugin_name = filename[:-3]  # Remove .py extension
                try:
                    self._load_plugin(plugin_name)
                except Exception as e:
                    logger.error(f"Failed to load plugin {plugin_name}: {e}")

    def _load_plugin(self, plugin_name: str):
        """Load a single plugin by name."""
        plugin_path = os.path.join(self.plugin_dir, f"{plugin_name}.py")

        # Check if plugin file exists
        if not os.path.exists(plugin_path):
            logger.error(f"Plugin file not found: {plugin_path}")
            return

        # Load plugin module
        try:
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)

            # Register plugin
            self.plugins[plugin_name] = {
                "module": plugin_module,
                "intents": getattr(plugin_module, "SUPPORTED_INTENTS", [plugin_name]),
                "handler": getattr(plugin_module, "handle", None),
            }
            logger.info(f"Loaded plugin: {plugin_name}")
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            raise

    def execute_plugin(self, intent: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a plugin based on intent.

        Args:
            intent (str): The intent to handle
            data (Dict[str, Any]): Data to pass to plugin

        Returns:
            Dict[str, Any]: Plugin execution result
        """
        # Find plugin for intent
        plugin_entry = None
        for plugin_name, plugin_info in self.plugins.items():
            if intent in plugin_info["intents"]:
                plugin_entry = plugin_info
                break

        if plugin_entry is None or not plugin_entry["handler"]:
            logger.warning(f"No plugin found for intent: {intent}")
            return {
                "success": False,
                "error": f"No handler for intent '{intent}'",
                "intent": intent,
            }

        # Execute plugin
        try:
            # Add memory to context
            context = {
                "short_term_memory": self.short_term_memory,
                "long_term_memory": self.long_term_memory,
            }

            # Call handler
            result = plugin_entry["handler"](data, context)

            # Store result in short-term memory
            self.short_term_memory.remember(
                f"Executed {intent} plugin with result: {result.get('message', 'No message')}",
                {"plugin": plugin_entry["module"].__name__, "intent": intent},
            )

            return result
        except Exception as e:
            logger.error(f"Plugin execution error: {e}")
            return {"success": False, "error": str(e), "intent": intent}

    def execute(
        self,
        plugin_name: str,
        intent_data: dict,
        user_id: str,
        tone: str = "neutral",
    ) -> str:
        """
        Executes a plugin with memory context integration.

        Args:
            plugin_name (str): The plugin identifier (module name).
            intent_data (dict): Parsed intent structure from NLP pipeline.
            user_id (str): Unique user/session identifier.
            tone (str): Optional stylistic tone for the plugin output.

        Returns:
            str: Plugin response text.
        """

        logger.info(f"[Orchestrator] Executing plugin: {plugin_name} for user: {user_id}")

        # Step 1: Retrieve contextual memory
        context = {
            "short_term": self.short_term_memory.retrieve(user_id),
            "long_term": self.long_term_memory.query_relevant(user_id, intent_data.get("raw_text", "")),
            "intent": intent_data,
        }

        try:
            # Step 2: Dispatch to plugin
            response = dispatch_task(plugin_name, params=intent_data, context=context, tone=tone)

            # Step 3: Update short-term memory
            self.short_term_memory.store(user_id, {
                "intent": intent_data.get("intent"),
                "entities": intent_data.get("entities"),
                "response": response
            })

            # Step 4: Persist to long-term memory (heuristic or plugin flag could be used)
            if plugin_name in {"reminder", "memory", "alarm"}:
                self.long_term_memory.store(user_id, {
                    "type": plugin_name,
                    "data": intent_data,
                    "output": response
                })

            return response

        except PluginExecutionError as e:
            logger.error(f"[Orchestrator] Plugin execution failed: {e}")
            return f"I'm afraid the {plugin_name} plugin encountered an error, My Lord."

        except Exception as e:
            logger.exception("[Orchestrator] Unexpected failure.")
            return f"An unexpected error occurred while executing '{plugin_name}', My Lord."


# Optional convenience entry point
orchestrator = PluginOrchestrator()
