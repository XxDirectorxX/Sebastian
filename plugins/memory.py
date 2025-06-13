
# plugins/memory.py

import logging

logger = logging.getLogger("Sebastian.Plugin.Memory")

SUPPORTED_INTENTS = ['store_memory', 'recall_memory']

async def handle(intent: dict) -> str:
    try:
        logger.info(f"[Memory] Handling intent: {intent}")
        # Example logic
        return "The memory has been recorded, My Lord."
    except Exception as e:
        logger.exception(f"[Memory] Failed to process intent.")
        return "My apologies, My Lord. That plugin encountered an error."
