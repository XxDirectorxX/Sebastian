
# core/personality/contextual_ai.py

import logging
from core.personality.persona_manager import get_context_modifiers

logger = logging.getLogger("Sebastian.ContextualAI")

def adjust_for_context(text: str, context: dict) -> str:
    modifiers = get_context_modifiers()
    irony = modifiers.get("irony_level", 0)
    formal = modifiers.get("formal", True)
    composure = modifiers.get("composure", 10)

    if irony > 3 and "error" in context.get("intent", ""):
        return f"Oh dear... another error. How utterly unexpected. {text}"
    if not formal:
        return text.replace("My Lord", "mate")  # Humorous fallback

    if composure < 4:
        return f"*sigh* {text}"
    return text
