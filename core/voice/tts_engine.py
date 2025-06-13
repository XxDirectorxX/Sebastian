# core/voice/tts_engine.py

import asyncio
import logging
import tempfile
import os

logger = logging.getLogger("Sebastian.TTS")

try:
    from TTS.api import TTS as CoquiTTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False

try:
    import piper_tts
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

CONFIG_PATH = "assistant_config.yaml"

class TTSEngine:
    def __init__(self, config):
        self.config = config
        self.voice_model = config.get("tts", {}).get("voice_model", "tts_models/en/ljspeech/tacotron2-DDC")
        self.fallback_model = config.get("tts", {}).get("fallback_model", "piper")
        self.device = "cuda" if config.get("tts", {}).get("use_gpu", False) else "cpu"

        self.tts = None
        self.fallback = None

        if COQUI_AVAILABLE:
            try:
                self.tts = CoquiTTS(model_name=self.voice_model, progress_bar=False, gpu=self.device == "cuda")
                logger.info(f"[TTS] Coqui TTS initialized with model: {self.voice_model}")
            except Exception as e:
                logger.error("[TTS] Coqui TTS failed to initialize.", exc_info=True)

        if self.tts is None and PIPER_AVAILABLE:
            try:
                self.fallback = piper_tts.PiperVoice.load("en_US-lessac-medium.onnx")
                logger.info("[TTS] Piper fallback loaded.")
            except Exception as e:
                logger.error("[TTS] Piper failed to load.", exc_info=True)

    async def synthesize(self, text: str) -> str:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                path = tmpfile.name

            if self.tts:
                logger.info("[TTS] Synthesizing with Coqui...")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.tts.tts_to_file, text, path)
            elif self.fallback:
                logger.info("[TTS] Synthesizing with Piper fallback...")
                audio_data = self.fallback.synthesize(text)
                with open(path, "wb") as f:
                    f.write(audio_data)
            else:
                logger.error("[TTS] No TTS engine available.")
                return ""

            return path

        except Exception as e:
            logger.exception("[TTS] Synthesis failed.")
            return ""
