
# plugins/alarm.py

import logging

logger = logging.getLogger("Sebastian.Plugin.Alarm")

SUPPORTED_INTENTS = ['set_alarm', 'cancel_alarm']

async def handle(intent: dict) -> str:
    try:
        logger.info(f"[Alarm] Handling intent: {intent}")
        # Example logic
        return "Your alarm has been set, My Lord."
    except Exception as e:
        logger.exception(f"[Alarm] Failed to process intent.")
        return "My apologies, My Lord. That plugin encountered an error."
