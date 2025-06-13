# core/voice/hotword_listener.py

import asyncio
import logging
import sounddevice as sd
import numpy as np
import queue
from threading import Thread

from core.voice.whisper_engine import transcribe_audio_async

logger = logging.getLogger("Sebastian.Hotword")

DEFAULT_HOTWORD = "sebastian, attend me"
AUDIO_QUEUE = queue.Queue()

class MicrophoneAccessError(Exception):
    pass


def _audio_callback(indata, frames, time, status):
    if status:
        logger.warning(f"Microphone status: {status}")
    AUDIO_QUEUE.put(indata.copy())


async def _record_audio(duration: float = 5.0, samplerate: int = 16000) -> np.ndarray:
    try:
        with sd.InputStream(samplerate=samplerate, channels=1, callback=_audio_callback):
            logger.info("Recording...")
            await asyncio.sleep(duration)
    except Exception as e:
        logger.error("Microphone access failed.", exc_info=True)
        raise MicrophoneAccessError("Unable to access microphone")

    frames = []
    while not AUDIO_QUEUE.empty():
        frames.append(AUDIO_QUEUE.get())
    audio = np.concatenate(frames, axis=0)
    return audio.flatten()


async def listen_loop(trigger_phrase: str = DEFAULT_HOTWORD, config: dict = None, fallback_cli: bool = False):
    logger.info("Starting hotword listener...")
    use_gpu = config.get("use_gpu", False) if config else False

    while True:
        try:
            logger.info("Awaiting hotword...")
            user_input = input("[You]: ").strip().lower()  # Simulated hotword via text
            if trigger_phrase in user_input:
                logger.info("Hotword detected. Recording user command...")
                audio_data = await _record_audio()
                transcription = await transcribe_audio_async(audio_data, use_gpu=use_gpu)
                return transcription

        except MicrophoneAccessError:
            if fallback_cli:
                logger.warning("Microphone unavailable. Switching to CLI fallback.")
                return input("[Fallback CLI] Enter your command: ")
            else:
                logger.error("Microphone failure with no CLI fallback.")
                return ""

        except Exception as e:
            logger.exception("Unexpected error in hotword listener.")
            await asyncio.sleep(2)
