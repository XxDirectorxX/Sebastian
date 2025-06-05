# core/plugins/greet.py

from core.plugin_utils import plugin_metadata

@plugin_metadata(name="greet", description="Delivers a personalized greeting", version="1.0")
def run(params: dict, context: dict = None, tone: str = None) -> str:
    context = context or {}
    user = context.get("user", "my lord")

    tone_prefix = {
        "cheerful": "A splendid day to you, ",
        "formal": "Greetings, ",
        "sarcastic": "Well, look who decided to show up — ",
    }.get(tone, "Good day, ")

    return f"{tone_prefix}{user}."

from core.plugins.plugin_manager import SebastianPlugin

class GreetPlugin(SebastianPlugin):
    def metadata(self):
        return {
            "name": "greet",
            "version": "1.0",
            "description": "Greets the user with a formal introduction.",
            "author": "Sebastian"
        }

    def run(self, **kwargs):
        title = kwargs.get("title", "My Lord")
        return f"Good evening, {title}. As always, I am at your service."
