import asyncio
import logging
import io
from typing import Optional
import numpy as np
from scipy.io.wavfile import write as scipy_write_wav # To convert numpy array to WAV bytes

try:
    from TTS.api import TTS as CoquiTTSAPI
except ImportError:
    CoquiTTSAPI = None # Gracefully handle if not installed

logger = logging.getLogger("Sebastian.SpeechSynthesizer")

class SpeechSynthesizerError(Exception):
    """Custom exception for SpeechSynthesizer errors."""
    pass

class SpeechSynthesizer:
    """
    Synthesizes speech from text using a Text-to-Speech engine (Coqui TTS).
    """
    def __init__(
        self,
        model_name: str = "tts_models/en/ljspeech/tacotron2-DDC", # A common default
        vocoder_name: Optional[str] = None, # Often bundled or inferred
        # progress_bar: bool = False, # Coqui TTS Synthesizer param
        device: Optional[str] = None # "cuda", "cpu", or None for auto-detect
    ):
        """
        Initializes the Coqui TTS speech synthesizer.

        :param model_name: Name of the Coqui TTS model to use.
        :param vocoder_name: Name of the Coqui TTS vocoder to use (if separate).
        :param device: Device to run the model on ("cuda" or "cpu").
        """
        if CoquiTTSAPI is None:
            msg = "Coqui TTS library is not installed. Please install it with 'pip install TTS'."
            logger.error(msg)
            raise SpeechSynthesizerError(msg)

        self.model_name = model_name
        self.vocoder_name = vocoder_name
        self.device = device
        self._tts_instance = None # Loaded lazily
        logger.info(
            f"SpeechSynthesizer configured with model: {self.model_name}, "
            f"vocoder: {self.vocoder_name or 'default/bundled'}, device: {self.device or 'auto'}"
        )

    async def _load_tts(self):
        """Loads the Coqui TTS instance. This can take time."""
        if self._tts_instance is None:
            logger.info(f"Loading Coqui TTS model '{self.model_name}'... This may take a moment.")
            try:
                # CoquiTTSAPI uses model_name and optionally vocoder_name, etc.
                # It handles device selection internally if not specified.
                assert CoquiTTSAPI is not None, "CoquiTTSAPI should have been checked in __init__"
                self._tts_instance = await asyncio.to_thread(
                    CoquiTTSAPI,
                    model_name=self.model_name,
                    vocoder_name=self.vocoder_name,
                    progress_bar=False, # Disable progress bar for cleaner logs
                    gpu=(self.device == "cuda") if self.device else None # True for CUDA, False for CPU, None for auto
                )
                logger.info(f"Coqui TTS model '{self.model_name}' loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Coqui TTS model '{self.model_name}': {e}", exc_info=True)
                raise SpeechSynthesizerError(f"Failed to load Coqui TTS model: {e}")
        return self._tts_instance

    async def synthesize_to_wav_bytes(self, text: str) -> Optional[bytes]:
        """
        Synthesizes speech from text and returns it as WAV audio bytes.

        :param text: The text to synthesize.
        :return: WAV audio data as bytes, or None if synthesis fails.
        """
        tts = await self._load_tts()
        if not tts:
            return None

        if not text.strip():
            logger.warning("Cannot synthesize empty text.")
            return None

        try:
            logger.info(f"Starting speech synthesis for text: \"{text[:50]}...\"")
            
            # The tts.tts_to_file method is simpler if we just need a file,
            # but tts.tts() returns a waveform (list of floats or numpy array)
            # which we can then convert to bytes.
            # waveform = await asyncio.to_thread(tts.tts, text, speaker=tts.speakers[0] if tts.speakers else None, language=tts.languages[0] if tts.languages else None)
            
            # Simpler approach if speaker/language args are not needed or handled by model default:
            waveform_np = await asyncio.to_thread(tts.tts, text)

            if not isinstance(waveform_np, np.ndarray):
                # Some models might return list, convert to numpy array
                waveform_np = np.array(waveform_np)
            
            # Ensure waveform is 1D
            if waveform_np.ndim > 1:
                waveform_np = waveform_np.squeeze() # Remove singleton dimensions if any
                if waveform_np.ndim > 1: # If still multi-dimensional, attempt to take first channel
                    logger.warning(f"Multi-channel audio detected (shape: {waveform_np.shape}), taking first channel.")
                    waveform_np = waveform_np[:, 0]


            # Coqui TTS output is typically float32. For WAV, int16 is common.
            # Normalize and convert to int16
            if np.issubdtype(waveform_np.dtype, np.floating):
                audio_int16 = (waveform_np * 32767).astype(np.int16)
            else: # If already int, assume it's in a good range or needs specific handling
                logger.warning(f"TTS output waveform is not float ({waveform_np.dtype}), direct conversion to int16 might be lossy or incorrect.")
                audio_int16 = waveform_np.astype(np.int16)


            sample_rate = tts.synthesizer.output_sample_rate if tts.synthesizer else 22050 # Fallback SR

            wav_bytes_io = io.BytesIO()
            # Use scipy to write wav data to BytesIO object
            await asyncio.to_thread(scipy_write_wav, wav_bytes_io, sample_rate, audio_int16)
            
            wav_bytes = wav_bytes_io.getvalue()
            logger.info(f"Speech synthesis complete. Generated {len(wav_bytes)} bytes of audio data.")
            return wav_bytes

        except Exception as e:
            logger.error(f"Error during speech synthesis: {e}", exc_info=True)
            return None

    def release(self):
        """Releases resources if necessary (Coqui TTS objects are usually managed by GC)."""
        if self._tts_instance:
            logger.info("Coqui TTS instance will be garbage collected.")
            # For some specific backends, there might be explicit cleanup, but generally not for TTS.api.TTS
            self._tts_instance = None

    def __del__(self):
        self.release()


# Example usage (for testing)
# async def _test_speech_synthesizer(synthesizer: SpeechSynthesizer):
#     test_text = "Hello, My Lord. This is a test of my vocal articulation."
#     logger.info(f"Testing speech synthesis with text: \"{test_text}\"")
    
#     wav_bytes = await synthesizer.synthesize_to_wav_bytes(test_text)
    
#     if wav_bytes:
#         logger.info(f"Successfully synthesized audio ({len(wav_bytes)} bytes).")
#         # To play it back (requires a playback library like sounddevice or simpleaudio):
#         try:
#             import sounddevice as sd
#             import soundfile as sf # To read the wav bytes
            
#             # Create a SoundFile object from bytes
#             sf_object = sf.SoundFile(io.BytesIO(wav_bytes))
#             data = sf_object.read(dtype='float32')
#             sample_rate = sf_object.samplerate
            
#             logger.info(f"Attempting to play synthesized audio (Sample Rate: {sample_rate})...")
#             await asyncio.to_thread(sd.play, data, sample_rate)
#             await asyncio.to_thread(sd.wait) # Wait until playback is finished
#             logger.info("Playback finished.")
            
#         except ImportError:
#             logger.warning("SoundDevice or SoundFile not installed. Cannot play audio. Please install with 'pip install sounddevice soundfile'.")
#         except Exception as e:
#             logger.error(f"Error playing audio: {e}", exc_info=True)
#     else:
#         logger.error("Speech synthesis test failed to produce audio bytes.")

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
#     async def main_test():
#         try:
#             # Ensure the model you choose is appropriate and will download.
#             # "tts_models/en/ljspeech/tacotron2-DDC" is a common one.
#             # "tts_models/en/vctk/vits" is another good option (multi-speaker, might need speaker_idx)
#             # Check `tts --list_models` for available models.
#             synthesizer = SpeechSynthesizer(
#                 model_name="tts_models/en/ljspeech/tacotron2-DDC", 
#                 device="cpu" # Use "cuda" if available and configured
#             )
#             await _test_speech_synthesizer(synthesizer)
#         except SpeechSynthesizerError as e:
#             logger.error(f"Synthesizer test initialization failed: {e}")
#         except Exception as e:
#             logger.error(f"Unexpected error in main_test: {e}", exc_info=True)

#     # asyncio.run(main_test())