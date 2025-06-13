import importlib
import inspect
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

registered_plugins: Dict[str, Callable] = {}
_plugin_modules: Dict[str, str] = {}  # Map plugin name to module name for reload tracking


def plugin(name: str, description: str, example_phrases: Optional[List[str]] = None) -> Callable:
    """
    Decorator to register a function as a plugin with metadata.

    Args:
        name (str): Unique plugin name.
        description (str): Description of the plugin.
        example_phrases (List[str], optional): Example phrases for intent training.

    Raises:
        ValueError: If the plugin name is already registered or function signature is invalid.

    Returns:
        Callable: The decorated function.
    """
    if example_phrases is None:
        example_phrases = []

    def decorator(func: Callable) -> Callable:
        # Check plugin signature: must accept exactly one parameter (str or untyped)
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if len(params) != 1:
            raise ValueError(
                f"Plugin '{name}' must have exactly one parameter (command: str), got {len(params)}"
            )
        param = params[0]
        if param.annotation not in (str, inspect._empty):
            raise ValueError(
                f"Plugin '{name}' parameter must be of type 'str' or untyped, got {param.annotation}"
            )

        if name in registered_plugins:
            raise ValueError(f"Plugin name '{name}' is already registered.")

        registered_plugins[name] = func
        _plugin_modules[name] = func.__module__
        func.plugin_metadata = {
            "name": name,
            "description": description,
            "example_phrases": example_phrases,
        }
        logger.info(f"Registered plugin '{name}' from module '{func.__module__}'")
        return func

    return decorator


def load_plugins(plugin_folder: str = "core/plugins") -> None:
    """
    Dynamically loads all plugin modules from the specified folder (non-recursive).

    Args:
        plugin_folder (str): Path to the folder containing plugin modules.

    Raises:
        FileNotFoundError: If the plugin folder does not exist or is not a directory.
        ImportError: If any plugin module fails to import.
    """
    plugins_path = Path(plugin_folder)
    if not plugins_path.exists() or not plugins_path.is_dir():
        raise FileNotFoundError(f"Plugin folder '{plugin_folder}' does not exist or is not a directory.")

    loaded_modules: Set[str] = set()

    for plugin_file in plugins_path.glob("*.py"):
        if plugin_file.name == "__init__.py":
            continue  # Skip package init file if present

        module_name = plugin_file.stem
        module_path = f"core.plugins.{module_name}"

        if module_path in loaded_modules:
            logger.debug(f"Plugin module '{module_path}' already loaded; skipping.")
            continue

        try:
            importlib.import_module(module_path)
            loaded_modules.add(module_path)
            logger.info(f"Loaded plugin module: {module_path}")
        except Exception as e:
            logger.error(f"Failed to load plugin module '{module_path}': {e}")
            raise ImportError(f"Failed to load plugin module '{module_path}'") from e

    # Verify at least one plugin registered from each loaded module
    registered_modules = {func.__module__ for func in registered_plugins.values()}
    missing_plugins = loaded_modules - registered_modules
    if missing_plugins:
        logger.warning(f"Modules loaded but no plugins registered from: {missing_plugins}")


def reload_plugin(name: str) -> bool:
    """
    Reloads a registered plugin's module by plugin name and refreshes plugin registration.

    Args:
        name (str): Plugin name to reload.

    Returns:
        bool: True if reload successful and plugin re-registered, False otherwise.
    """
    if name not in registered_plugins:
        logger.warning(f"Plugin '{name}' not registered; cannot reload.")
        return False

    module_name = _plugin_modules.get(name)
    if not module_name:
        logger.warning(f"No module information for plugin '{name}'; cannot reload.")
        return False

    try:
        module = importlib.import_module(module_name)
        importlib.reload(module)
        logger.info(f"Reloaded plugin module '{module_name}' for plugin '{name}'.")

        # Clear old registrations from this module
        to_remove = [pname for pname, func in registered_plugins.items() if func.__module__ == module_name]
        for pname in to_remove:
            del registered_plugins[pname]
            _plugin_modules.pop(pname, None)

        # Re-import the module to re-register plugins
        importlib.import_module(module_name)
        logger.info(f"Re-registered plugins from module '{module_name}' after reload.")
        return True

    except Exception as e:
        logger.error(f"Error reloading plugin '{name}': {e}")
        return False
