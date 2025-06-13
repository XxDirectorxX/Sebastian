
# core/voice/wake_word_detector.py

import asyncio
import logging
import sounddevice as sd
import numpy as np
import vosk
import queue

logger = logging.getLogger("Sebastian.WakeWordDetector")

MODEL_PATH = "models/wakeword"
WAKE_WORD = "sebastian"

class WakeWordDetector:
    def __init__(self, model_path=MODEL_PATH, wake_word=WAKE_WORD):
        self.model = vosk.Model(model_path)
        self.wake_word = wake_word.lower()
        self.q = queue.Queue()

    def _callback(self, indata, frames, time, status):
        if status:
            logger.warning(status)
        self.q.put(bytes(indata))

    async def listen_for_wakeword(self):
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=self._callback):
            rec = vosk.KaldiRecognizer(self.model, 16000)
            logger.info("[WakeWord] Listening for: " + self.wake_word)

            while True:
                data = self.q.get()
                if rec.AcceptWaveform(data):
                    result = rec.Result().lower()
                    if self.wake_word in result:
                        logger.info("[WakeWord] Triggered by: " + result)
                        return True
                await asyncio.sleep(0.01)
