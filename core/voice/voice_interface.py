# core/voice/voice_interface.py

import asyncio
import logging
import sounddevice as sd
import numpy as np
import io
import soundfile as sf # For loading ack/error sounds
from typing import Callable, Awaitable, Optional, List, Tuple, Any, Dict

from .wake_word_detector import WakeWordDetector
from .speech_recognizer import SpeechRecognizer
from .speech_synthesizer import SpeechSynthesizer

logger = logging.getLogger("Sebastian.VoiceInterface")

class VoiceInterfaceError(Exception):
    """Custom exception for VoiceInterface errors."""
    pass

class VoiceInterfaceState:
    IDLE = "IDLE"
    LISTENING_FOR_WAKE_WORD = "LISTENING_FOR_WAKE_WORD"
    RECORDING_COMMAND = "RECORDING_COMMAND"
    PROCESSING_COMMAND = "PROCESSING_COMMAND" # (STT, NLP, TTS)
    SPEAKING = "SPEAKING"

class VoiceInterface:
    """
    Manages voice interaction: wake word, STT, TTS, and audio I/O.
    """
    def __init__(
        self,
        wake_word_detector: WakeWordDetector,
        speech_recognizer: SpeechRecognizer,
        speech_synthesizer: SpeechSynthesizer,
        config: Dict[str, Any],
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.wake_word_detector = wake_word_detector
        self.speech_recognizer = speech_recognizer
        self.speech_synthesizer = speech_synthesizer
        self.config = config
        self._loop = loop or asyncio.get_event_loop()

        self.sample_rate = self.config.get("sample_rate", 16000)
        self.channels = self.config.get("channels", 1)
        self.input_device_index = self.config.get("input_device_index")
        self.output_device_index = self.config.get("output_device_index")
        self.command_recording_duration_s = self.config.get("command_recording_duration_seconds", 5.0)
        
        self.ack_sound_path = self.config.get("ack_sound_path")
        self.error_sound_path = self.config.get("error_sound_path")
        self._ack_sound_data: Optional[Tuple[np.ndarray, int]] = None
        self._error_sound_data: Optional[Tuple[np.ndarray, int]] = None

        self._audio_queue = asyncio.Queue(maxsize=100) # Store chunks of audio frames
        self._stop_event = asyncio.Event()
        self._processing_task: Optional[asyncio.Task] = None
        self._input_stream: Optional[sd.InputStream] = None
        
        self.current_state = VoiceInterfaceState.IDLE
        self._command_processor_callback: Optional[Callable[[str], Awaitable[Optional[str]]]] = None
        self._command_audio_buffer: List[bytes] = []

        self._load_feedback_sounds()
        logger.info("VoiceInterface initialized.")

    def _load_feedback_sounds(self):
        if self.ack_sound_path:
            try:
                data, sr = sf.read(self.ack_sound_path, dtype='float32')
                self._ack_sound_data = (data, sr)
                logger.info(f"Loaded acknowledgment sound from: {self.ack_sound_path}")
            except Exception as e:
                logger.error(f"Failed to load acknowledgment sound {self.ack_sound_path}: {e}")
        if self.error_sound_path:
            try:
                data, sr = sf.read(self.error_sound_path, dtype='float32')
                self._error_sound_data = (data, sr)
                logger.info(f"Loaded error sound from: {self.error_sound_path}")
            except Exception as e:
                logger.error(f"Failed to load error sound {self.error_sound_path}: {e}")

    async def _play_feedback_sound(self, sound_data: Optional[Tuple[np.ndarray, int]]):
        if sound_data:
            data, sr = sound_data
            try:
                # Ensure output device is set if specified
                current_output_device = sd.default.device[1] # Default output
                if self.output_device_index is not None:
                    sd.default.device = (sd.default.device[0], self.output_device_index)

                await asyncio.to_thread(sd.play, data, sr)
                await asyncio.to_thread(sd.wait)
                logger.debug("Feedback sound played.")
            except Exception as e:
                logger.error(f"Error playing feedback sound: {e}", exc_info=True)
            finally:
                # Reset to default output device if it was changed
                if self.output_device_index is not None:
                     sd.default.device = (sd.default.device[0], current_output_device)


    def _audio_callback(self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
        """Callback for sounddevice InputStream. Runs in a separate thread."""
        if status:
            logger.warning(f"Audio input status: {status}")
        if self._stop_event.is_set():
            return

        try:
            # Ensure indata is bytes. Porcupine expects int16 samples, but we'll pass bytes.
            # The wake_word_detector.process_audio_frame expects bytes (raw PCM frame)
            # and it should be of length self.wake_word_detector.frame_length * 2 (for int16)
            # indata is usually float32 from sounddevice, convert to int16 bytes
            
            # Convert float32 to int16
            indata_int16 = (indata * 32767).astype(np.int16)
            self._audio_queue.put_nowait(indata_int16.tobytes())
        except asyncio.QueueFull:
            logger.warning("Audio queue is full. Dropping audio frame.")
        except Exception as e:
            logger.error(f"Error in audio callback: {e}", exc_info=True)


    async def _process_audio_stream(self):
        logger.info("Audio processing stream started.")
        self.current_state = VoiceInterfaceState.LISTENING_FOR_WAKE_WORD
        
        frames_for_command_recording = int(self.sample_rate * self.command_recording_duration_s / self.wake_word_detector.frame_length)
        # This calculation needs to be based on how many chunks we get from queue vs. raw frame_length
        # Let's assume each item from queue is one Porcupine frame for simplicity here.
        # A more robust way is to count bytes or duration.
        
        expected_frame_bytes = self.wake_word_detector.frame_length * 2 # 2 bytes per int16 sample

        while not self._stop_event.is_set():
            try:
                audio_chunk_bytes = await self._audio_queue.get()
                self._audio_queue.task_done()

                if not audio_chunk_bytes or len(audio_chunk_bytes) < expected_frame_bytes:
                    # logger.debug(f"Skipping small audio chunk: {len(audio_chunk_bytes)} bytes")
                    continue
                
                # Ensure we only process chunks of the exact size Porcupine expects
                # This might mean buffering smaller chunks or splitting larger ones.
                # For now, assume audio_chunk_bytes is a single processable frame for Porcupine.
                # This part needs careful alignment with how audio_callback provides data.
                # If audio_callback provides chunks larger/smaller than frame_length, this needs adjustment.
                # For now, let's assume it's aligned.

                if self.current_state == VoiceInterfaceState.LISTENING_FOR_WAKE_WORD:
                    # We need to ensure audio_chunk_bytes is exactly one frame for Porcupine
                    # If audio_callback puts larger chunks, we need to iterate through them.
                    # For simplicity, let's assume audio_chunk_bytes IS a single frame.
                    # This is a strong assumption and likely needs refinement.
                    # A better way: audio_callback puts fixed-size chunks (e.g. 1024 samples).
                    # _process_audio_stream then feeds these to Porcupine frame by frame.
                    
                    # Simplified: Assume audio_chunk_bytes is a single Porcupine frame
                    if len(audio_chunk_bytes) == expected_frame_bytes:
                        keyword_index = self.wake_word_detector.process_audio_frame(audio_chunk_bytes)
                        if keyword_index is not None and keyword_index >= 0:
                            logger.info("Wake word detected!")
                            await self._play_feedback_sound(self._ack_sound_data)
                            self.current_state = VoiceInterfaceState.RECORDING_COMMAND
                            self._command_audio_buffer.clear()
                            # Start a timer or frame count for recording
                            self._frames_recorded_for_command = 0
                            self._target_frames_for_command = int(
                                self.sample_rate * self.command_recording_duration_s
                            ) # Total samples
                    else:
                        # This case indicates mismatch between callback chunk size and Porcupine frame size.
                        # logger.warning(f"Audio chunk size {len(audio_chunk_bytes)} mismatch with Porcupine frame size {expected_frame_bytes}")
                        # This part needs robust handling of chunking for Porcupine.
                        # For now, we'll just append to buffer if recording.
                        pass


                elif self.current_state == VoiceInterfaceState.RECORDING_COMMAND:
                    self._command_audio_buffer.append(audio_chunk_bytes)
                    # Calculate total samples in buffer
                    current_samples_in_buffer = sum(len(b)//2 for b in self._command_audio_buffer)

                    if current_samples_in_buffer >= self._target_frames_for_command:
                        logger.info(f"Command recording finished. Samples: {current_samples_in_buffer}")
                        self.current_state = VoiceInterfaceState.PROCESSING_COMMAND
                        
                        full_audio_bytes = b"".join(self._command_audio_buffer)
                        self._command_audio_buffer.clear()
                        
                        # Offload the actual processing to avoid blocking this loop
                        asyncio.create_task(self._handle_detected_command(full_audio_bytes))
                        # After dispatching, return to listening for wake word
                        self.current_state = VoiceInterfaceState.LISTENING_FOR_WAKE_WORD


            except asyncio.CancelledError:
                logger.info("Audio processing stream cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in audio processing stream: {e}", exc_info=True)
                # Reset state to avoid getting stuck
                self.current_state = VoiceInterfaceState.LISTENING_FOR_WAKE_WORD
                await asyncio.sleep(0.1) # Brief pause before retrying

        logger.info("Audio processing stream stopped.")

    async def _handle_detected_command(self, audio_buffer_bytes: bytes):
        if not self._command_processor_callback:
            logger.error("Command processor callback not set.")
            self.current_state = VoiceInterfaceState.LISTENING_FOR_WAKE_WORD
            return

        try:
            # 1. Transcribe audio
            logger.info("Transcribing command...")
            # SpeechRecognizer needs sample_rate, sample_width (2 for int16), channels
            transcribed_text = await self.speech_recognizer.transcribe_audio_bytes(
                audio_bytes=audio_buffer_bytes,
                sample_rate=self.sample_rate,
                sample_width=2, # Assuming int16 from our buffer
                channels=self.channels
            )

            if transcribed_text:
                logger.info(f"Transcribed text: '{transcribed_text}'")
                # 2. Process command
                response_text = await self._command_processor_callback(transcribed_text)
                
                if response_text:
                    logger.info(f"Response from system: '{response_text}'")
                    # 3. Synthesize response
                    self.current_state = VoiceInterfaceState.SPEAKING
                    wav_bytes = await self.speech_synthesizer.synthesize_to_wav_bytes(response_text)
                    if wav_bytes:
                        # 4. Play synthesized speech
                        await self.play_audio_bytes(wav_bytes, self.speech_synthesizer._tts_instance.synthesizer.output_sample_rate if self.speech_synthesizer._tts_instance else self.sample_rate)
                    else:
                        logger.error("Failed to synthesize speech for response.")
                        await self._play_feedback_sound(self._error_sound_data)
                else:
                    logger.info("No response generated by system or command not understood.")
                    # Optionally play a "not understood" sound or say something generic
            else:
                logger.info("Transcription failed or resulted in empty text.")
                await self._play_feedback_sound(self._error_sound_data)

        except Exception as e:
            logger.error(f"Error handling detected command: {e}", exc_info=True)
            await self._play_feedback_sound(self._error_sound_data)
        finally:
            # Always return to listening for wake word after processing attempt
            self.current_state = VoiceInterfaceState.LISTENING_FOR_WAKE_WORD


    async def play_audio_bytes(self, audio_data_bytes: bytes, sample_rate: int):
        """Plays WAV audio bytes."""
        try:
            logger.info(f"Playing audio ({len(audio_data_bytes)} bytes, SR: {sample_rate})...")
            # sounddevice.play expects a NumPy array. Convert bytes (assuming WAV format)
            # This is a simplified playback. If audio_data_bytes is raw PCM, it's easier.
            # If it's WAV, we need to parse it or ensure SpeechSynthesizer gives raw PCM + SR.
            # For now, assume SpeechSynthesizer gives WAV bytes, so we read it.
            
            current_output_device = sd.default.device[1]
            if self.output_device_index is not None:
                sd.default.device = (sd.default.device[0], self.output_device_index)

            with io.BytesIO(audio_data_bytes) as bio:
                data, sr_from_file = await asyncio.to_thread(sf.read, bio, dtype='float32')
            
            # It's better if synthesizer provides raw PCM and its sample rate directly.
            # For now, we trust sr_from_file if it's WAV, or use provided sample_rate for raw PCM.
            # Let's assume SpeechSynthesizer's output SR is what we should use.
            
            await asyncio.to_thread(sd.play, data, sample_rate) # Use the SR from synthesizer
            await asyncio.to_thread(sd.wait)
            logger.info("Audio playback finished.")
        except Exception as e:
            logger.error(f"Error playing audio bytes: {e}", exc_info=True)
        finally:
            if self.output_device_index is not None:
                sd.default.device = (sd.default.device[0], current_output_device)


    async def start_interaction_loop(self, command_processor_callback: Callable[[str], Awaitable[Optional[str]]]):
        if self._processing_task and not self._processing_task.done():
            logger.warning("Interaction loop already running.")
            return

        self._command_processor_callback = command_processor_callback
        self._stop_event.clear()
        
        # Determine blocksize for InputStream. Should align with Porcupine's frame_length.
        # Porcupine's frame_length is in samples.
        blocksize_samples = self.wake_word_detector.frame_length 
        # This ensures the callback gets chunks of the size Porcupine expects.

        try:
            self._input_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.input_device_index,
                dtype='float32', # Porcupine will convert from this if needed, or we convert in callback
                blocksize=blocksize_samples, # Feed Porcupine frame by frame
                callback=self._audio_callback
            )
            self._input_stream.start()
            logger.info(f"Audio input stream started. Device: {self._input_stream.device}, SR: {self.sample_rate}, Blocksize: {blocksize_samples} samples.")
            
            self._processing_task = self._loop.create_task(self._process_audio_stream())
            self.current_state = VoiceInterfaceState.LISTENING_FOR_WAKE_WORD
            logger.info("Voice interaction loop started. Listening for wake word...")
            
        except Exception as e:
            logger.error(f"Failed to start voice interaction loop: {e}", exc_info=True)
            if self._input_stream:
                self._input_stream.close()
            self._input_stream = None
            self.current_state = VoiceInterfaceState.IDLE
            raise VoiceInterfaceError(f"Failed to start audio stream: {e}")

    async def stop_interaction_loop(self):
        logger.info("Attempting to stop voice interaction loop...")
        self._stop_event.set()
        
        if self._input_stream:
            try:
                self._input_stream.stop()
                self._input_stream.close()
                logger.info("Audio input stream stopped and closed.")
            except Exception as e:
                logger.error(f"Error closing audio input stream: {e}", exc_info=True)
            self._input_stream = None

        if self._processing_task and not self._processing_task.done():
            logger.info("Cancelling audio processing task...")
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                logger.info("Audio processing task successfully cancelled.")
            except Exception as e:
                logger.error(f"Error during processing task cancellation: {e}", exc_info=True)
        self._processing_task = None
        
        # Clear the queue to prevent old data processing on restart
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        self.current_state = VoiceInterfaceState.IDLE
        logger.info("Voice interaction loop stopped.")

    def __del__(self):
        if not self._stop_event.is_set():
            # This might be called from a non-async context if object is GC'd
            # Best effort cleanup. Proper shutdown should be explicit via stop_interaction_loop.
            logger.warning("VoiceInterface deleted without explicit stop. Attempting cleanup.")
            if self._input_stream and self._input_stream.active:
                self._input_stream.stop()
                self._input_stream.close()
            # Cannot easily await async stop here.
