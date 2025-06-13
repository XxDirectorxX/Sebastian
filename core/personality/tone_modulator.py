
# core/personality/tone_modulator.py

import logging
from core.personality.persona_manager import get_tone_rules

logger = logging.getLogger("Sebastian.ToneModulator")

def determine_tone(intent: str, text: str, context: dict) -> str:
    rules = get_tone_rules()
    if not rules:
        return "neutral"

    # Prioritize intent-specific tone
    if intent in rules:
        tone = rules[intent]
        logger.info(f"[ToneModulator] Tone '{tone}' selected for intent '{intent}'")
        return tone

    sentiment = context.get("sentiment", "neutral")
    tone = rules.get(sentiment, "neutral")
    logger.info(f"[ToneModulator] Tone '{tone}' selected for sentiment '{sentiment}'")
    return tone
