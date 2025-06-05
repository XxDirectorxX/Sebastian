import asyncio
import logging
import tempfile
import os
import numpy as np
import soundfile as sf # For reading/writing audio files and converting byte data
from typing import Optional

try:
    import whisper
except ImportError:
    whisper = None # Gracefully handle if not installed

logger = logging.getLogger("Sebastian.SpeechRecognizer")

class SpeechRecognizerError(Exception):
    """Custom exception for SpeechRecognizer errors."""
    pass

class SpeechRecognizer:
    """
    Transcribes speech to text using the Whisper model.
    """
    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = "en",
        device: Optional[str] = None # "cuda", "cpu", or None for auto-detect
    ):
        """
        Initializes the Whisper speech recognizer.

        :param model_size: Size of the Whisper model to use (e.g., "tiny", "base", "small").
        :param language: Language of the speech. If None, Whisper will attempt to detect.
        :param device: Device to run the model on ("cuda" or "cpu").
        """
        if whisper is None:
            msg = "openai-whisper library is not installed. Please install it with 'pip install openai-whisper'."
            logger.error(msg)
            raise SpeechRecognizerError(msg)

        self.model_size = model_size
        self.language = language
        self.device = device
        self._model = None # Loaded lazily or on first use to avoid long init times if not immediately needed
        logger.info(f"SpeechRecognizer configured with model: {self.model_size}, lang: {self.language}, device: {self.device}")

    async def _load_model(self):
        """Loads the Whisper model. This can take time."""
        if self._model is None:
            if whisper is None:
                msg = "openai-whisper library is not installed. Please install it with 'pip install openai-whisper'."
                logger.error(msg)
                raise SpeechRecognizerError(msg)
            logger.info(f"Loading Whisper model '{self.model_size}'... This may take a moment.")
            try:
                self._model = await asyncio.to_thread(whisper.load_model, self.model_size, device=self.device)
                logger.info(f"Whisper model '{self.model_size}' loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Whisper model '{self.model_size}': {e}", exc_info=True)
                raise SpeechRecognizerError(f"Failed to load Whisper model: {e}")
        return self._model

    async def transcribe_audio_bytes(self, audio_bytes: bytes, sample_rate: int, sample_width: int, channels: int) -> Optional[str]:
        """
        Transcribes audio data provided as bytes.

        :param audio_bytes: Raw audio data.
        :param sample_rate: Sample rate of the audio.
        :param sample_width: Sample width in bytes (e.g., 2 for 16-bit).
        :param channels: Number of audio channels.
        :return: Transcribed text string, or None if transcription fails.
        """
        model = await self._load_model()
        if not model:
            return None

        # Whisper's transcribe method can take a NumPy array directly.
        # We need to convert the raw bytes to a float32 NumPy array.
        # Whisper expects mono audio, so if stereo, we might need to average or take one channel.
        
        try:
            # Determine the correct dtype based on sample_width
            if sample_width == 1: # 8-bit
                dtype = np.int8
            elif sample_width == 2: # 16-bit
                dtype = np.int16
            elif sample_width == 4: # 32-bit
                dtype = np.int32
            else:
                logger.error(f"Unsupported sample_width: {sample_width}")
                return None

            # Convert bytes to NumPy array
            audio_np = np.frombuffer(audio_bytes, dtype=dtype)

            # Reshape if multi-channel and convert to mono float32, normalized
            if channels > 1:
                audio_np = audio_np.reshape(-1, channels)
                audio_np = audio_np.mean(axis=1) # Convert to mono by averaging channels

            # Normalize to [-1.0, 1.0] if it's integer type
            if np.issubdtype(audio_np.dtype, np.integer):
                audio_np = audio_np.astype(np.float32) / np.iinfo(dtype).max
            elif not np.issubdtype(audio_np.dtype, np.floating): # If it's some other non-float type
                 logger.error(f"Audio data is not in a convertible integer or float format: {audio_np.dtype}")
                 return None


            logger.debug(f"Audio data converted to NumPy array. Shape: {audio_np.shape}, Dtype: {audio_np.dtype}, SR: {sample_rate}")

            # Perform transcription in a separate thread
            transcribe_options = {}
            if self.language:
                transcribe_options['language'] = self.language
            
            # Set fp16=False if device is CPU and model is large, as it can cause issues.
            # For smaller models or GPU, fp16 is generally fine and faster.
            if self.device == "cpu" and self.model_size in ["large", "large-v2", "large-v3"]:
                transcribe_options['fp16'] = False
            
            logger.info("Starting audio transcription with Whisper...")
            result = await asyncio.to_thread(model.transcribe, audio_np, **transcribe_options)
            
            text_result = result.get("text", "")
            if isinstance(text_result, list):
                transcribed_text = " ".join(text_result).strip()
            else:
                transcribed_text = str(text_result).strip()
            detected_lang = result.get("language", "unknown")
            logger.info(f"Transcription complete. Detected language: {detected_lang}. Text: '{transcribed_text}'")
            
            return transcribed_text

        except Exception as e:
            logger.error(f"Error during audio transcription: {e}", exc_info=True)
            return None

    async def transcribe_audio_file(self, audio_file_path: str) -> Optional[str]:
        """
        Transcribes an audio file.

        :param audio_file_path: Path to the audio file.
        :return: Transcribed text string, or None if transcription fails.
        """
        model = await self._load_model()
        if not model:
            return None
        
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return None

        try:
            logger.info(f"Starting transcription of audio file: {audio_file_path}...")
            transcribe_options = {}
            if self.language:
                transcribe_options['language'] = self.language
            if self.device == "cpu" and self.model_size in ["large", "large-v2", "large-v3"]:
                transcribe_options['fp16'] = False

            result = await asyncio.to_thread(model.transcribe, audio_file_path, **transcribe_options)
            
            text_result = result.get("text", "")
            if isinstance(text_result, list):
                transcribed_text = " ".join(text_result).strip()
            else:
                transcribed_text = str(text_result).strip()
            detected_lang = result.get("language", "unknown")
            logger.info(f"File transcription complete. Detected language: {detected_lang}. Text: '{transcribed_text}'")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Error transcribing audio file '{audio_file_path}': {e}", exc_info=True)
            return None

# Example usage (for testing)
# async def _test_speech_recognizer_with_bytes(recognizer: SpeechRecognizer):
#     # This requires actual audio byte data.
#     # For a real test, you'd capture audio from a microphone.
#     # Create a dummy silent audio for structure testing:
#     sample_rate = 16000  # Whisper prefers 16kHz
#     duration = 2  # seconds
#     channels = 1
#     sample_width = 2 # 16-bit
#     num_samples = int(sample_rate * duration)
#     # Create silent audio (zeros)
#     silent_audio_bytes = b'\x00\x00' * num_samples * channels
    
#     logger.info("Testing transcription with dummy silent audio bytes...")
#     text = await recognizer.transcribe_audio_bytes(silent_audio_bytes, sample_rate, sample_width, channels)
#     if text is not None:
#         logger.info(f"Test transcription from bytes (silent audio): '{text}' (expected to be empty or minimal)")
#     else:
#         logger.error("Test transcription from bytes failed.")

# async def _test_speech_recognizer_with_file(recognizer: SpeechRecognizer):
#     # Create a dummy WAV file for testing
#     # You would replace this with a path to an actual audio file for a real test.
#     sample_rate = 16000
#     duration = 3
#     frequency = 440 # A4 note
#     t = np.linspace(0, duration, int(sample_rate * duration), False)
#     audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
#     # Ensure it's float32 for soundfile write, then Whisper will handle it
#     audio_data_float32 = audio_data.astype(np.float32)

#     temp_file_path = ""
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
#             temp_file_path = tmpfile.name
#         sf.write(temp_file_path, audio_data_float32, sample_rate)
        
#         logger.info(f"Testing transcription with dummy WAV file: {temp_file_path}")
#         text = await recognizer.transcribe_audio_file(temp_file_path)
#         if text is not None:
#             logger.info(f"Test transcription from file: '{text}'")
#         else:
#             logger.error("Test transcription from file failed.")
#     except Exception as e:
#         logger.error(f"Error in file test setup: {e}")
#     finally:
#         if temp_file_path and os.path.exists(temp_file_path):
#             os.remove(temp_file_path)


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#     # Ensure ffmpeg is installed and in PATH if not using a direct WAV file.
    
#     async def main_test():
#         try:
#             # Using a small model for quicker testing; ensure it's downloaded.
#             recognizer = SpeechRecognizer(model_size="tiny.en", device="cpu") 
#             # await _test_speech_recognizer_with_bytes(recognizer) # Test with byte data
#             await _test_speech_recognizer_with_file(recognizer) # Test with a generated file
#         except SpeechRecognizerError as e:
#             logger.error(f"Recognizer test initialization failed: {e}")
#         except Exception as e:
#             logger.error(f"Unexpected error in main_test: {e}", exc_info=True)

#     # asyncio.run(main_test())