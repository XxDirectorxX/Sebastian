# core/plugin_orchestrator.py

import logging
from core.task_dispatcher import dispatch_task, PluginNotFoundError, PluginExecutionError
from core.memory import recall_recent_context, store_execution_result
from core.context_manager import update_context
from core.personality_simulation import adapt_tone
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PluginExecutionErrorWrapper(Exception):
    """Wrapper for plugin execution failures at orchestrator level."""
    pass

class PluginOrchestrator:
    def __init__(self):
        self.context = recall_recent_context()

    def execute_plan(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Executes a series of plugin actions defined in the intent.
        Handles fallbacks, sequencing, and emotional context propagation.
        """
        results = []

        actions = intent.get("actions", [])
        tone = intent.get("tone", "neutral")
        priority = intent.get("priority", "normal")

        logger.info(f"Executing {len(actions)} plugin(s) with tone={tone}, priority={priority}.")

        for index, action in enumerate(actions):
            plugin_name = action.get("plugin")
            params = action.get("params", {})

            logger.debug(f"Dispatching plugin '{plugin_name}' with params {params}.")

            try:
                personality_hint = adapt_tone(tone, plugin_name, params)
                result = dispatch_task(plugin_name, params, context=self.context, tone=personality_hint)

                logger.info(f"Plugin '{plugin_name}' executed successfully. Result: {result}")
                results.append({
                    "plugin": plugin_name,
                    "result": result,
                    "success": True
                })

                update_context(plugin_name, result)
                store_execution_result(plugin_name, params, result, success=True)

            except PluginNotFoundError as e:
                logger.error(f"Plugin '{plugin_name}' not found: {str(e)}")
                results.append({
                    "plugin": plugin_name,
                    "result": str(e),
                    "success": False
                })
                # Continue executing next plugins despite missing plugin
                continue

            except PluginExecutionError as e:
                logger.error(f"Plugin '{plugin_name}' execution failed: {str(e)}")
                results.append({
                    "plugin": plugin_name,
                    "result": str(e),
                    "success": False
                })
                store_execution_result(plugin_name, params, str(e), success=False)
                # Halt execution on plugin failure
                raise PluginExecutionErrorWrapper(f"Execution halted at '{plugin_name}' due to error.") from e

            except Exception as e:
                logger.exception(f"Unexpected error during plugin '{plugin_name}' execution: {str(e)}")
                results.append({
                    "plugin": plugin_name,
                    "result": str(e),
                    "success": False
                })
                raise PluginExecutionErrorWrapper(f"Execution halted at '{plugin_name}' due to unexpected error.") from e

        return results
