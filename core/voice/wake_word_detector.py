import logging
import os
from typing import List, Optional, Union

try:
    import pvporcupine
    from pvporcupine import PorcupineError
except ImportError:
    pvporcupine = None # Gracefully handle if not installed, though it's a requirement
    PorcupineError = Exception  # Fallback to generic Exception if import fails

logger = logging.getLogger("Sebastian.WakeWordDetector")

class WakeWordDetectorError(Exception):
    """Custom exception for WakeWordDetector errors."""
    pass

class WakeWordDetector:
    """
    Detects a wake word using the Porcupine engine.
    """
    def __init__(
        self,
        access_key: str,
        keyword_paths: Union[str, List[str]],
        sensitivities: Union[float, List[float]] = 0.5,
        library_path: Optional[str] = None,
        model_path: Optional[str] = None
    ):
        """
        Initializes the Porcupine wake word detector.

        :param access_key: AccessKey obtained from PicoVoice Console.
        :param keyword_paths: Absolute path(s) to Porcupine keyword files (.ppn).
        :param sensitivities: Sensitivity for detecting keywords. A higher sensitivity reduces miss rate
                              at the cost of increased false alarm rate. Value should be within [0, 1].
        :param library_path: Absolute path to Porcupine's dynamic library.
        :param model_path: Absolute path to Porcupine's model file.
        """
        if pvporcupine is None:
            msg = "pvporcupine library is not installed. Please install it with 'pip install pvporcupine'."
            logger.error(msg)
            raise WakeWordDetectorError(msg)

        if not access_key:
            msg = "Porcupine AccessKey is missing. Please provide it in the configuration."
            logger.error(msg)
            raise WakeWordDetectorError(msg)

        if isinstance(keyword_paths, str):
            keyword_paths = [keyword_paths]
        if isinstance(sensitivities, (float, int)):
            sensitivities = [float(sensitivities)] * len(keyword_paths)

        if not keyword_paths:
            msg = "No keyword_paths provided for Porcupine."
            logger.error(msg)
            raise WakeWordDetectorError(msg)
            
        resolved_keyword_paths = []
        for path in keyword_paths:
            if not os.path.exists(path):
                # Attempt to resolve relative to a common assets directory if not absolute
                # This is a heuristic; ideally, paths in config should be absolute or clearly relative.
                # For now, we assume paths from config are either absolute or resolvable as is.
                logger.warning(f"Keyword file not found at '{path}'. Ensure the path is correct.")
                # raise WakeWordDetectorError(f"Keyword file not found: {path}") # Or handle more gracefully
            resolved_keyword_paths.append(path)


        try:
            self._porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=resolved_keyword_paths, # Use resolved paths
                sensitivities=sensitivities,
                library_path=library_path,
                model_path=model_path
            )
            self.sample_rate = self._porcupine.sample_rate
            self.frame_length = self._porcupine.frame_length
            logger.info(
                f"Porcupine wake word detector initialized. "
                f"Sample Rate: {self.sample_rate}, Frame Length: {self.frame_length}"
            )
        except PorcupineError as e:
            logger.error(f"Failed to initialize Porcupine: {e}", exc_info=True)
            raise WakeWordDetectorError(f"Porcupine initialization failed: {e}")
            logger.error(f"Failed to initialize Porcupine: {e}", exc_info=True)
            raise WakeWordDetectorError(f"Porcupine initialization failed: {e}")

    def process_audio_frame(self, pcm_frame: bytes) -> Optional[int]:
        """
        Processes a frame of audio data.

        :param pcm_frame: A frame of audio data (16-bit linear PCM).
                          The length must be `self.frame_length`.
        :return: Index of the detected keyword if wake word is detected, otherwise None.
                 Returns -1 if an error occurs during processing.
        """
        if len(pcm_frame) // 2 != self.frame_length: # PCM is 16-bit, so 2 bytes per sample
            logger.warning(f"Invalid audio frame size. Expected {self.frame_length * 2} bytes, got {len(pcm_frame)} bytes.")
            return None # Or raise error

        # Convert bytes to list of int16 samples as required by porcupine.process()
        # This assumes the input pcm_frame is already in the correct format (bytes)
        # and Porcupine handles the conversion internally if it expects a list of shorts.
        # The pvporcupine.process method expects a list/tuple of int16 audio samples.
        import struct
        try:
            audio_frame_shorts = struct.unpack('%dh' % self.frame_length, pcm_frame)
        except struct.error as e:
            logger.error(f"Error unpacking PCM frame: {e}. Frame length: {len(pcm_frame)}, Expected shorts: {self.frame_length}")
            return -1 # Indicate error

        if not hasattr(self, '_porcupine') or self._porcupine is None:
            logger.error("Porcupine instance is not initialized or has been released.")
            return -1  # Indicate error

        try:
            keyword_index = self._porcupine.process(audio_frame_shorts)
            if keyword_index >= 0:
                logger.info(f"Wake word detected! Keyword index: {keyword_index}")
        except PorcupineError as e:
            logger.error(f"Error processing audio frame with Porcupine: {e}", exc_info=True)
            return -1 # Indicate error
            logger.error(f"Error processing audio frame with Porcupine: {e}", exc_info=True)
            return -1 # Indicate error
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error during Porcupine processing: {e}", exc_info=True)
            return -1


    def release(self):
        """
        Releases resources used by Porcupine.
        """
        if hasattr(self, '_porcupine') and self._porcupine is not None:
            try:
                self._porcupine.delete()
            except PorcupineError as e:
                logger.error(f"Error releasing Porcupine resources: {e}", exc_info=True)

    def __del__(self):
        self.release()

# Example usage (for testing, not part of the class's primary role in the system)
# async def _test_wake_word_detector():
#     # This requires a live audio stream and proper configuration
#     # For now, this is a conceptual test
#     try:
#         # Ensure you have a valid AccessKey and .ppn file
#         # Create a dummy .ppn file or use a real one for testing if available
#         # For this example, we'll assume config is correctly set up elsewhere
#         # and audio frames are being fed.
#         config = {
#             "access_key": "YOUR_PICOVOICE_ACCESS_KEY", # Replace
#             "keyword_paths": ["path/to/your/sebastian.ppn"], # Replace
#             "sensitivity": 0.7
#         }
#         detector = WakeWordDetector(
#             access_key=config["access_key"],
#             keyword_paths=config["keyword_paths"],
#             sensitivities=config["sensitivity"]
#         )
#         logger.info("Test detector initialized. Waiting for audio frames (simulated)...")
#         # In a real test, you'd feed audio frames from a microphone here.
#         # e.g., using PyAudio:
#         # import pyaudio
#         # pa = pyaudio.PyAudio()
#         # audio_stream = pa.open(...)
#         # while True:
#         #     pcm_frame = audio_stream.read(detector.frame_length) # Read bytes
#         #     result = detector.process_audio_frame(pcm_frame)
#         #     if result is not None and result >=0:
#         #         print("Wake word detected in test!")
#         #         break
#     except WakeWordDetectorError as e:
#         logger.error(f"Test failed: {e}")
#     except Exception as e:
#         logger.error(f"Unexpected error in test: {e}", exc_info=True)
#     finally:
#         if 'detector' in locals():
#             detector.release()

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#     # asyncio.run(_test_wake_word_detector()) # Requires actual audio input setup