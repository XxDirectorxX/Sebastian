# core/voice/voice_output.py

import asyncio
import logging
import os
import platform
import subprocess
import time

try:
    import simpleaudio as sa
    SIMPLEAUDIO_AVAILABLE = True
except ImportError:
    SIMPLEAUDIO_AVAILABLE = False

logger = logging.getLogger("Sebastian.VoiceOutput")


async def speak_text(audio_path: str):
    if not os.path.exists(audio_path):
        logger.error(f"[VoiceOutput] Audio file does not exist: {audio_path}")
        return

    logger.info(f"[VoiceOutput] Speaking from: {audio_path}")
    start_time = time.time()

    try:
        if SIMPLEAUDIO_AVAILABLE:
            logger.info("[VoiceOutput] Using simpleaudio backend.")
            wave_obj = sa.WaveObject.from_wave_file(audio_path)
            play_obj = wave_obj.play()
            while play_obj.is_playing():
                await asyncio.sleep(0.1)

        else:
            system = platform.system()
            if system == "Windows":
                subprocess.Popen(["start", "", audio_path], shell=True)
            elif system == "Darwin":
                subprocess.Popen(["afplay", audio_path])
            elif system == "Linux":
                subprocess.Popen(["aplay", audio_path])
            else:
                logger.error("[VoiceOutput] Unsupported platform for audio playback.")
                return
            await asyncio.sleep(5)  # crude delay to allow audio playback

        duration = time.time() - start_time
        logger.info(f"[VoiceOutput] Playback complete. Duration: {duration:.2f}s")

    except Exception as e:
        logger.exception("[VoiceOutput] Failed to play audio.")
