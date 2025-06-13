import os
import sys
import importlib.util
import inspect
import asyncio
import logging
from typing import Dict, Any, Callable, Union, Optional, List, Coroutine, Awaitable, Type
from abc import ABC, abstractmethod

logger = logging.getLogger("Sebastian.PluginManager")

class SebastianPlugin(ABC):
    """
    Abstract base for all plugins.
    Plugins may override async run() for async operation.
    """
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return plugin metadata dictionary with keys: name, description, version."""
        pass

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Optional plugin initialization lifecycle hook."""
        pass

    def shutdown(self) -> None:
        """Optional plugin shutdown lifecycle hook."""
        pass

    def run(self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None, tone: Optional[str] = None) -> Union[str, Awaitable[str]]:
        """
        Run plugin synchronously or asynchronously.
        Return a string response or coroutine resolving to string.
        """
        raise NotImplementedError("Plugin must implement run method.")


def plugin_metadata(name: str, description: str, version: str = "1.0"):
    """
    Decorator to add metadata to function-based plugins.
    """
    def decorator(func):
        func.plugin_meta = {
            "name": name,
            "description": description,
            "version": version
        }
        return func
    return decorator


class PluginLoadError(Exception):
    pass


class PluginManager:
    def __init__(self, plugin_directory: str, config: Optional[Dict[str, Any]] = None):
        self.plugin_directory = plugin_directory
        self.plugins: Dict[str, Union[SebastianPlugin, Callable]] = {}
        self.config = config or {}
        self._load_errors: List[str] = []

    def _validate_metadata(self, meta: Dict[str, Any]) -> bool:
        required_keys = {"name", "description", "version"}
        if not all(key in meta for key in required_keys):
            logger.error(f"Plugin metadata missing required keys: {meta}")
            return False
        if not isinstance(meta['name'], str) or not meta['name']:
            logger.error(f"Plugin metadata 'name' must be non-empty string: {meta}")
            return False
        return True

    def load_plugins(self, whitelist: Optional[List[str]] = None, blacklist: Optional[List[str]] = None) -> None:
        """
        Discover and load plugins from directory.
        Can filter plugins by whitelist/blacklist.
        """
        if not os.path.isdir(self.plugin_directory):
            logger.error(f"Plugin directory does not exist or is not a directory: {self.plugin_directory}")
            return

        for filename in os.listdir(self.plugin_directory):
            if not filename.endswith(".py") or filename.startswith("__"):
                continue

            module_name = filename[:-3]
            if whitelist and module_name not in whitelist:
                logger.debug(f"Skipping plugin '{module_name}' not in whitelist.")
                continue
            if blacklist and module_name in blacklist:
                logger.debug(f"Skipping plugin '{module_name}' due to blacklist.")
                continue

            file_path = os.path.join(self.plugin_directory, filename)
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    raise PluginLoadError(f"Cannot load spec for {module_name}")

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                logger.info(f"Plugin module '{module_name}' loaded.")

                self._register_plugins_from_module(module)

            except Exception as e:
                error_msg = f"Failed to load plugin '{module_name}': {e}"
                self._load_errors.append(error_msg)
                logger.error(error_msg, exc_info=True)

    def _register_plugins_from_module(self, module) -> None:
        """
        Register both class-based and function-based plugins from module.
        """
        # Class-based
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, SebastianPlugin) and obj is not SebastianPlugin:
                try:
                    instance = obj()
                    meta = instance.metadata()
                    if not self._validate_metadata(meta):
                        raise PluginLoadError(f"Invalid metadata in plugin class '{obj.__name__}': {meta}")
                    instance.initialize(self.config.get(meta['name'], {}))
                    self.plugins[meta['name']] = instance
                    logger.info(f"Registered class-based plugin '{meta['name']}'")
                except Exception as e:
                    logger.error(f"Failed to initialize plugin class '{obj.__name__}': {e}", exc_info=True)

        # Function-based
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if hasattr(obj, "plugin_meta"):
                meta = getattr(obj, "plugin_meta")
                if not self._validate_metadata(meta):
                    logger.error(f"Invalid metadata in function plugin '{name}': {meta}")
                    continue
                self.plugins[meta['name']] = obj
                logger.info(f"Registered function-based plugin '{meta['name']}'")

    def get_plugin(self, name: str) -> Optional[Union[SebastianPlugin, Callable]]:
        return self.plugins.get(name)

    async def execute_plugin(self, name: str, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None, tone: Optional[str] = None) -> Optional[str]:
        plugin = self.get_plugin(name)
        if not plugin:
            logger.error(f"Plugin '{name}' not found.")
            raise ValueError(f"Plugin '{name}' not found.")

        try:
            if callable(plugin):
                result = plugin(params=params, context=context, tone=tone)  # function plugin
            else:
                result = plugin.run(params=params, context=context, tone=tone)  # class plugin

            if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                return await result
            return result
        except Exception as e:
            logger.error(f"Error executing plugin '{name}': {e}", exc_info=True)
            return None

    def shutdown_all(self) -> None:
        """
        Call shutdown hooks on all loaded plugins.
        """
        for plugin in self.plugins.values():
            try:
                if isinstance(plugin, SebastianPlugin):
                    plugin.shutdown()
                    logger.info(f"Shutdown plugin '{plugin.metadata().get('name', 'unknown')}'")
            except Exception as e:
                logger.error(f"Error during shutdown of plugin '{plugin}': {e}")

    @property
    def load_errors(self) -> List[str]:
        return self._load_errors
