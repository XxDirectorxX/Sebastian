# core/plugins/device_control.py

from core.plugin_utils import plugin_metadata

@plugin_metadata(name="device_control", description="Controls smart home devices", version="1.0")
def run(params: dict, context: dict = None, tone: str = None) -> str:
    context = context or {}
    device = params.get("device")
    action = params.get("action")

    if not device or not action:
        return "I need both a device and an action to proceed."

    prefix = {
        "commanding": "Executing order: ",
        "playful": "On it! ",
        "formal": "Very well. "
    }.get(tone, "")

    return f"{prefix}{action.capitalize()}ing the {device} now."

