# core/plugins/plugin_manager.py

import os
import importlib.util
import inspect
from typing import Dict, Any, Callable, Union
from core.plugin_utils import plugin_metadata
from abc import ABC, abstractmethod

class SebastianPlugin(ABC):
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def run(self, params: dict, context: dict = None, tone: str = None) -> str:
        pass


def plugin_metadata(name: str, description: str, version: str = "1.0"):
    def decorator(func):
        func.plugin_meta = {
            "name": name,
            "description": description,
            "version": version
        }
        return func
    return decorator


class PluginManager:
    def __init__(self, plugin_directory: str):
        self.plugin_directory = plugin_directory
        self.plugins: Dict[str, Union[SebastianPlugin, Callable]] = {}

    def load_plugins(self):
        for filename in os.listdir(self.plugin_directory):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3]
                file_path = os.path.join(self.plugin_directory, filename)
                
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Load class-based plugins
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, SebastianPlugin) and obj is not SebastianPlugin:
                        instance = obj()
                        meta = instance.metadata()
                        self.plugins[meta['name']] = instance

                # Load function-based plugins
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    if hasattr(obj, 'plugin_meta'):
                        meta = obj.plugin_meta
                        self.plugins[meta['name']] = obj

    def get_plugin(self, name: str) -> Union[SebastianPlugin, Callable, None]:
        return self.plugins.get(name)

    def execute_plugin(self, name: str, **kwargs) -> Any:
        plugin = self.get_plugin(name)
        if plugin:
            try:
                if callable(plugin):  # Function-based
                    return plugin(**kwargs)
                else:  # Class-based
                    return plugin.run(**kwargs)
            except Exception as e:
                print(f"[PluginManager] Error executing plugin '{name}': {e}")
        else:
            raise ValueError(f"Plugin '{name}' not found.")
