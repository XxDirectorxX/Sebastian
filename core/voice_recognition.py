import logging
import whisper
import pyaudio
import numpy as np
import queue
import threading
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class VoiceRecognition:
    def __init__(self, model_size="base", device="cpu", sample_rate=16000, chunk_size=1024):
        self.model_size = model_size
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        logger.info(f"Loading Whisper model: {model_size} on device: {device}")
        self.model = whisper.load_model(self.model_size, device=self.device)

        self.audio_interface = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue()
        self.running = False
        self.transcription = ""
        self._lock = threading.Lock()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def start_stream(self):
        if self.stream is not None:
            logger.warning("Stream already running")
            return

        logger.debug("Starting audio stream...")
        self.stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        self.running = True
        self.stream.start_stream()

    def stop_stream(self):
        if self.stream:
            logger.debug("Stopping audio stream...")
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.running = False

    def _collect_audio(self, timeout=5):
        frames = []
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                data = self.audio_queue.get(timeout=timeout - (time.time() - start_time))
                frames.append(np.frombuffer(data, np.int16))
            except queue.Empty:
                break
        if frames:
            audio_np = np.concatenate(frames)
            return audio_np
        return np.array([])

    def transcribe(self, timeout=5):
        if not self.running:
            logger.error("Stream not running, cannot transcribe")
            return ""

        audio_np = self._collect_audio(timeout=timeout)
        if audio_np.size == 0:
            logger.info("No audio data collected during timeout period")
            return ""

        try:
            logger.debug("Running Whisper transcription...")
            result = self.model.transcribe(audio_np, fp16=False)
            text = result.get("text", "").strip()
            logger.info(f"Transcription result: {text}")
            with self._lock:
                self.transcription = text
            return text
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return ""

    def get_latest_transcription(self):
        with self._lock:
            return self.transcription

    def terminate(self):
        logger.debug("Terminating VoiceRecognition instance")
        self.stop_stream()
        self.audio_interface.terminate()
