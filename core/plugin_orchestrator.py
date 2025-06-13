# core/plugin_orchestrator.py

import os
import json
import logging
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import asyncio

from core.memory.short_term import ShortTermMemory
from core.memory.long_term import LongTermMemory
from core.task_dispatcher import execute_plugin_intent
from core.plugin_manager import SessionManager

logger = logging.getLogger("Sebastian.PluginOrchestrator")
logger.setLevel(logging.DEBUG)


class PluginOrchestrator:
    def __init__(
        self,
        plugin_dir: str = "plugins",
        session_manager: Optional[SessionManager] = None,
        short_term_memory: Optional[ShortTermMemory] = None,
        long_term_memory: Optional[LongTermMemory] = None,
    ):
        self.plugin_dir = plugin_dir
        self.plugins: Dict[str, Dict[str, Any]] = {}
        self.session_manager = session_manager
        self.short_term_memory = short_term_memory or ShortTermMemory()
        self.long_term_memory = long_term_memory or LongTermMemory()

        self._discover_plugins()

    def _discover_plugins(self):
        if not os.path.isdir(self.plugin_dir):
            logger.warning(f"Plugin directory not found: {self.plugin_dir}")
            return

        for filename in os.listdir(self.plugin_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                path = os.path.join(self.plugin_dir, filename)
                plugin_name = filename[:-3]
                spec = importlib.util.spec_from_file_location(plugin_name, path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "SUPPORTED_INTENTS") and hasattr(module, "handle_intent"):
                        for intent in module.SUPPORTED_INTENTS:
                            self.plugins[intent] = {
                                "name": plugin_name,
                                "module": module,
                            }
                        logger.info(f"Loaded plugin: {plugin_name} ({module.SUPPORTED_INTENTS})")
                    else:
                        logger.warning(f"Plugin {plugin_name} missing required fields.")

    async def route_intent(self, intent_data: dict, user_id: str, tone: str = "neutral") -> str:
        intent = intent_data.get("intent")
        args = intent_data.get("entities", [])

        plugin_entry = self.plugins.get(intent)
        if not plugin_entry:
            logger.warning(f"No plugin registered for intent: {intent}")
            return f"I'm afraid I do not recognize the command '{intent}', My Lord."

        module = plugin_entry["module"]
        context = {
            "short_term": self.short_term_memory.retrieve(user_id),
            "long_term": self.long_term_memory.query_relevant(user_id, intent_data.get("raw_text", "")),
            "session": self.session_manager.context.get("sessions", {}).get(user_id, {})
        }

        try:
            result = await execute_plugin_intent(module, intent, args, context)
            self.short_term_memory.store(user_id, {
                "intent": intent,
                "args": args,
                "response": result
            })
            return result
        except Exception as e:
            logger.exception("Plugin execution failed")
            return f"An error occurred while executing '{intent}', My Lord."


_orchestrator_instance: Optional[PluginOrchestrator] = None

def get_plugin_orchestrator(**kwargs) -> PluginOrchestrator:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = PluginOrchestrator(**kwargs)
    return _orchestrator_instance
