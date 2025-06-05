# core/plugins/session.py

from core.plugin_utils import plugin_metadata

@plugin_metadata(name="session", description="Manages session-related commands", version="1.0")
def run(params: dict, context: dict = None, tone: str = None) -> str:
    action = params.get("action")

    if action == "end":
        return "Session terminated. I remain at your service, should you summon me again."

    if action == "reset":
        context.clear()
        return "All session data has been cleared, my lord."

    return "Specify whether to 'end' or 'reset' the session."
