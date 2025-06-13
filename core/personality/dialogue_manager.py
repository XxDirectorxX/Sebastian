
# core/personality/dialogue_manager.py

import logging
from core.personality.tone_modulator import determine_tone
from core.personality.mannerisms import apply_mannerism
from core.personality.contextual_ai import adjust_for_context
from core.personality.persona_manager import get_persona_setting

logger = logging.getLogger("Sebastian.DialogueManager")

class DialogueManager:
    def __init__(self, persona="Sebastian Michaelis"):
        self.persona = persona

    def generate_response(self, intent: str, user_input: str, context: dict) -> str:
        try:
            base_reply = self._intent_to_base_response(intent, user_input)
            tone = determine_tone(intent, user_input, context)
            styled_reply = apply_mannerism(base_reply, tone)
            contextualized = adjust_for_context(styled_reply, context)
            return contextualized
        except Exception as e:
            logger.exception("[DialogueManager] Failed to generate response.")
            return get_persona_setting("error_responses.generic")

    def _intent_to_base_response(self, intent: str, user_input: str) -> str:
        # Primitive fallback mapping — should be replaced by plugin/dialogue corpus
        default_map = {
            "greet": "Good day, My Lord.",
            "farewell": "Until next time, My Lord.",
            "confirm": "As you command.",
            "error": "My sincerest apologies, My Lord.",
            "unknown": "I'm afraid I do not understand."
        }
        return default_map.get(intent, f"I have received your instruction: '{user_input}'")
