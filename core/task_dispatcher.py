# core/task_dispatcher.py

import importlib
import logging
from typing# core/task_dispatcher.py

import asyncio
import logging

logger = logging.getLogger("Sebastian.TaskDispatcher")


class PluginExecutionError(Exception):
    pass


async def execute_plugin_intent(module, intent: str, args: list, context: dict) -> str:
    try:
        if hasattr(module, "handle_intent"):
            func = module.handle_intent
            if asyncio.iscoroutinefunction(func):
                return await func(intent, args, context)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, intent, args, context)
        else:
            raise PluginExecutionError("Plugin lacks 'handle_intent' function")

    except Exception as e:
        logger.exception(f"Failed to execute intent '{intent}' in plugin '{module.__name__}'")
        raise PluginExecutionError(f"Execution failed for intent '{intent}': {e}")
 import Any, Dict, Callable, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PLUGIN_NAMESPACE = "core.plugins"

class PluginNotFoundError(Exception):
    pass

class PluginExecutionError(Exception):
    pass

_plugin_cache: Dict[str, Any] = {}

def dispatch_task(
    plugin_name: str,
    params: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    tone: Optional[str] = None,
) -> str:
    """
    Dynamically imports and executes a plugin's main task function.

    Args:
        plugin_name (str): Name of the plugin module (without `.py`).
        params (dict): Parameters to pass to the plugin function.
        context (dict, optional): Execution context/memory for this user session.
        tone (str, optional): Tone modifier (e.g., 'polite', 'urgent', 'sarcastic').

    Returns:
        str: The plugin’s response.

    Raises:
        PluginNotFoundError: If plugin module cannot be imported.
        PluginExecutionError: If plugin execution fails or 'run' entrypoint missing.
    """
    try:
        logger.debug(f"Dispatching plugin: {plugin_name}")

        if plugin_name in _plugin_cache:
            plugin_module = _plugin_cache[plugin_name]
        else:
            module_path = f"{PLUGIN_NAMESPACE}.{plugin_name}"
            try:
                plugin_module = importlib.import_module(module_path)
                _plugin_cache[plugin_name] = plugin_module
            except ModuleNotFoundError as e:
                logger.error(f"Plugin '{plugin_name}' not found in namespace.")
                raise PluginNotFoundError(f"Plugin '{plugin_name}' not found.") from e

        if not hasattr(plugin_module, "run") or not callable(plugin_module.run):
            raise PluginExecutionError(f"Plugin '{plugin_name}' missing required 'run' callable.")

        result = plugin_module.run(params=params, context=context or {}, tone=tone)

        if not isinstance(result, str):
            result = str(result)

        logger.debug(f"Plugin '{plugin_name}' executed successfully. Result: {result}")

        return result

    except (PluginNotFoundError, PluginExecutionError):
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in plugin '{plugin_name}': {e}")
        raise PluginExecutionError(f"Error executing plugin '{plugin_name}': {e}") from e
