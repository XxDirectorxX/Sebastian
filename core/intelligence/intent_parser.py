# core/intelligence/intent_parser.py

import logging
from core.intelligence.ner_engine import extract_entities

logger = logging.getLogger("Sebastian.IntentParser")

# Example intent map; in production, this may be dynamically loaded from plugin registry
INTENT_KEYWORDS = {
    "greet": ["hello", "hi", "greetings"],
    "weather": ["weather", "forecast", "temperature"],
    "time": ["time", "clock", "hour"],
    "fallback": []
}


def parse_intent(user_text: str) -> dict:
    logger.debug(f"Parsing intent from input: {user_text}")

    intent_match = "fallback"
    confidence = 0.0
    args = []

    text_lower = user_text.lower()
    matched = False

    for intent, keywords in INTENT_KEYWORDS.items():
        for word in keywords:
            if word in text_lower:
                intent_match = intent
                confidence = 0.9  # heuristic; replace with ML confidence if model-driven
                matched = True
                break
        if matched:
            break

    # Extract arguments using NER engine if available
    try:
        entities = extract_entities(user_text)
        args = [v for k, v in entities.items()]
    except Exception as e:
        logger.warning("NER extraction failed", exc_info=True)

    return {
        "intent": intent_match,
        "entities": args,
        "confidence": confidence,
    }
