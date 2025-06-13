
# plugins/greet.py

import logging

logger = logging.getLogger("Sebastian.Plugin.Greet")

SUPPORTED_INTENTS = ['greet_user']

async def handle(intent: dict) -> str:
    try:
        logger.info(f"[Greet] Handling intent: {intent}")
        # Example logic
        return "Good day, My Lord. How may I serve?"
    except Exception as e:
        logger.exception(f"[Greet] Failed to process intent.")
        return "My apologies, My Lord. That plugin encountered an error."
