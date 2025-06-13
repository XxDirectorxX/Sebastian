# core/orchestrator/plugin_orchestrator.py

import importlib
import os
import inspect

class PluginOrchestrator:
    def __init__(self, config, memory, personality):
        self.config = config
        self.memory = memory
        self.personality = personality
        self.plugins = self.load_plugins()

    def load_plugins(self):
        plugin_dir = "plugins"
        plugins = {}

        for filename in os.listdir(plugin_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3]
                module = importlib.import_module(f"{plugin_dir}.{module_name}")

                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj) and hasattr(obj, "intent"):
                        plugins[obj.intent] = obj

        print(f"[PluginOrchestrator] Loaded plugins: {list(plugins.keys())}")
        return plugins

    async def dispatch(self, intent):
        if intent in self.plugins:
            try:
                response = await self._run_plugin(self.plugins[intent])
                return response
            except Exception as e:
                return f"An error occurred while executing '{intent}': {e}"
        else:
            return f"I do not recognize the command '{intent}'."

    async def _run_plugin(self, plugin_fn):
        if inspect.iscoroutinefunction(plugin_fn):
            return await plugin_fn()
        else:
            return plugin_fn()
