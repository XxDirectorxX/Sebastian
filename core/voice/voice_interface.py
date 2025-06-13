
# core/voice/voice_interface.py

import asyncio
import logging
from core.voice.wake_word_detector import WakeWordDetector
from core.voice.speech_synthesis import say
from core.voice.whisper_engine import transcribe_audio
from core.voice.tts_engine import synthesize_text
from core.voice.voice_output import speak_text

logger = logging.getLogger("Sebastian.VoiceInterface")

class VoiceInterface:
    def __init__(self):
        self.wake_detector = WakeWordDetector()

    async def run(self):
        while True:
            await self.wake_detector.listen_for_wakeword()
            await say("Yes, My Lord.", emotion="neutral")

            try:
                result = await transcribe_audio()
                logger.info(f"[VoiceInterface] Transcribed: {result}")
                return result
            except Exception as e:
                logger.exception("[VoiceInterface] Failed during transcription.")
