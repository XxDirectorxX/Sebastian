
# core/voice/speech_synthesis.py

import logging
from core.voice.tts_engine import synthesize_text
from core.voice.voice_output import speak_text
from core.voice.emotion_shaper import get_emotion_config

logger = logging.getLogger("Sebastian.SpeechSynthesis")

async def say(text: str, emotion: str = "neutral"):
    try:
        config = get_emotion_config(emotion)
        audio_path = synthesize_text(text, config=config)
        await speak_text(audio_path)
    except Exception as e:
        logger.exception(f"[SpeechSynthesis] Failed to say: {text}")
