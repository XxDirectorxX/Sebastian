# main.py

import asyncio
import logging
import json
from pathlib import Path

from core.config.config_loader import load_assistant_config
from core.memory.short_term import ShortTermMemory
from core.memory.long_term import LongTermMemory
from core.memory.vector_store import init_vector_store
from core.personality.persona_manager import PersonaManager
from core.plugin_orchestrator import get_plugin_orchestrator
from core.voice.hotword_listener import start_hotword_listener
from core.voice.voice_output import speak_text
from core.voice.whisper_engine import transcribe_audio
from core.context_manager import SessionManager
from core.intelligence.intent_parser import parse_intent

logger = logging.getLogger("Sebastian.Main")
logging.basicConfig(level=logging.INFO)

async def initialize_context(path: Path):
    """Ensure context file exists and is well-formed."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"sessions": {}}, f)
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError:
            logger.warning("Context file corrupted. Resetting.")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"sessions": {}}, f)

async def main():
    logger.info("🧠 Initializing Sebastian AI Daemon...")

    try:
        config = load_assistant_config("assistant_config.yaml")
        user_id = config.get("default_user", "user123")

        short_term_memory = ShortTermMemory()
        long_term_memory = LongTermMemory()
        init_vector_store(config.get("vector_store", {}))

        persona = PersonaManager("assets/personas/persona_profile.yaml")

        await initialize_context(Path("core/context/context.json"))

        plugin_orchestrator = get_plugin_orchestrator()
        session_manager = SessionManager(user_id)

        logger.info("🎤 Starting hotword listener loop...")
        await start_hotword_listener(
            whisper_callback=transcribe_audio,
            plugin_orchestrator=plugin_orchestrator,
            session=session_manager,
            persona=persona
        )

    except KeyboardInterrupt:
        logger.info("✋ Keyboard interrupt received. Shutting down gracefully.")
    except Exception as e:
        logger.exception("❌ Fatal error in daemon boot sequence.")

if __name__ == "__main__":
    asyncio.run(main())
