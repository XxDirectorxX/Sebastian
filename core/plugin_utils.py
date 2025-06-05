# core/plugin_utils.py

import importlib
import inspect
import logging
from pathlib import Path
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)

registered_plugins: Dict[str, Callable] = {}

def plugin(name: str, description: str, example_phrases: List[str] = None) -> Callable:
    """
    Decorator to register a function as a plugin with metadata.

    Args:
        name (str): Plugin name.
        description (str): Description of the plugin.
        example_phrases (List[str], optional): Example phrases for intent training.

    Returns:
        Callable: The decorated function.
    """
    if example_phrases is None:
        example_phrases = []

    def decorator(func: Callable) -> Callable:
        # Check plugin signature: must accept one argument (command: str)
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if len(params) != 1 or params[0].annotation not in (str, inspect._empty):
            logger.warning(f"Plugin '{name}' should have exactly one argument of type str")

        registered_plugins[name] = func
        func.plugin_metadata = {
            "name": name,
            "description": description,
            "example_phrases": example_phrases,
        }
        logger.info(f"Registered plugin '{name}'")
        return func

    return decorator


def load_plugins(plugin_folder: str = "core/plugins") -> None:
    """
    Dynamically loads all plugin modules from the specified folder.

    Args:
        plugin_folder (str): Path to the folder containing plugin modules.
    """
    plugins_path = Path(plugin_folder)
    if not plugins_path.exists() or not plugins_path.is_dir():
        logger.error(f"Plugin folder '{plugin_folder}' does not exist or is not a directory.")
        return

    for plugin_file in plugins_path.glob("*.py"):
        module_name = plugin_file.stem
        module_path = f"core.plugins.{module_name}"
        try:
            importlib.import_module(module_path)
            logger.info(f"Loaded plugin module: {module_name}")
        except Exception as e:
            logger.error(f"Failed to load plugin '{module_name}': {e}")


def reload_plugin(name: str) -> bool:
    """
    Reloads a registered plugin module by name.

    Args:
        name (str): Plugin name to reload.

    Returns:
        bool: True if reload successful, False otherwise.
    """
    if name not in registered_plugins:
        logger.warning(f"Plugin '{name}' not registered; cannot reload.")
        return False

    try:
        module = inspect.getmodule(registered_plugins[name])
        if module:
            importlib.reload(module)
            logger.info(f"Reloaded plugin '{name}'.")
            return True
        else:
            logger.warning(f"Cannot find module for plugin '{name}'.")
            return False
    except Exception as e:
        logger.error(f"Error reloading plugin '{name}': {e}")
        return False
