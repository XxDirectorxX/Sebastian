# core/plugins/reminder.py

from core.plugin_utils import plugin_metadata

@plugin_metadata(name="reminder", description="Creates a time-based reminder", version="1.0")
def run(params: dict, context: dict = None, tone: str = None) -> str:
    task = params.get("task")
    time = params.get("time")

    if not task or not time:
        return "I need both a task and a time to set a reminder."

    flair = {
        "motivational": "Let us crush procrastination.",
        "serious": "No detail shall be forgotten.",
        "light": "Sticky notes are for peasants."
    }.get(tone, "")

    return f"Reminder set: '{task}' at {time}. {flair}"

