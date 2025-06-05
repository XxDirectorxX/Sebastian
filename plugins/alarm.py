# core/plugins/alarm.py

from core.plugin_utils import plugin_metadata

@plugin_metadata(name="alarm", description="Sets an alarm for a specified time", version="1.0")
def run(params: dict, context: dict = None, tone: str = None) -> str:
    context = context or {}
    time = params.get("time")

    if not time:
        return "I require a specific time to set the alarm, my lord."

    tone_suffix = {
        "urgent": "I shall ensure you're alerted without fail.",
        "casual": "Got it. No worries.",
        "respectful": "As you wish."
    }.get(tone, "Very well.")

    return f"Alarm set for {time}. {tone_suffix}"
