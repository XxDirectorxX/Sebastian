# core/plugins/memory.py

from core.plugin_utils import plugin_metadata

@plugin_metadata(name="memory", description="Stores or retrieves information from memory", version="1.0")
def run(params: dict, context: dict = None, tone: str = None) -> str:
    context = context or {}
    action = params.get("action")
    key = params.get("key")
    value = params.get("value")

    if action == "store" and key and value:
        context.setdefault("memory", {})[key] = value
        return f"Stored '{key}' as '{value}', my lord."

    elif action == "retrieve" and key:
        memory = context.get("memory", {})
        result = memory.get(key)
        if result:
            return f"The value of '{key}' is '{result}', as requested."
        else:
            return f"I found nothing stored under '{key}', my lord."

    return "Specify whether to store or retrieve, along with a key."
