"""
Asynchronous voice interaction system for Sebastian assistant.

Handles continuous voice processing with wake word detection,
speech recognition, and response generation.
"""
import asyncio
import logging
import os
import queue
import threading
import time
from typing import Dict, Any, Optional, Set, Callable, List
import wave

import numpy as np
import sounddevice as sd
import pvporcupine
import whisper
from scipy.io import wavfile

from core.orchestrator.async_system_integrator import get_async_system_integrator
from core.voice.speech_synthesis import SpeechSynthesizer

logger = logging.getLogger(__name__)

class VoiceInteractionSystem:
    """
    Manages continuous voice interaction with wake word detection,
    speech recognition, and response synthesis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize voice interaction system.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self._loop = asyncio.get_event_loop()
        
        # System state
        self.is_listening = False
        self.is_responding = False
        self.is_processing = False
        self._stop_event = asyncio.Event()
        self._audio_queue = asyncio.Queue(maxsize=100)
        self._wake_word_detected = asyncio.Event()
        self._commands: Set[str] = set()
        
        # Voice processing parameters
        self.sample_rate = 16000
        self.frame_length = 512
        self.buffer_seconds = 5  # Keep 5 seconds of audio buffer
        self.silence_threshold = 0.03
        self.silence_seconds = 1.0  # Stop recording after 1s of silence
        
        # Initialize wake word detector
        self._init_wake_word_detector()
        
        # Initialize speech recognizer
        self._init_speech_recognizer()
        
        # Initialize speech synthesizer
        self.speech_synthesizer = SpeechSynthesizer(
            voice_id="sebastian",
            model_path="assets/voice_models/sebastian_voice_model.pth"
        )
        
        # Audio buffers
        self._audio_buffer = np.array([], dtype=np.float32)
        self._current_recording = np.array([], dtype=np.float32)
        
        # Command handlers
        self._command_handlers = {}
        
        logger.info("Voice interaction system initialized")
    
    def _init_wake_word_detector(self):
        """Initialize wake word detection."""
        try:
            # Initialize Porcupine wake word detector
            self.porcupine = pvporcupine.create(
                access_key=os.environ.get("PICOVOICE_KEY", ""),
                keywords=["sebastian"],
                sensitivities=[0.7]
            )
            logger.info("Wake word detector initialized")
        except Exception as e:
            logger.error(f"Error initializing wake word detector: {e}")
            self.porcupine = None
    
    def _init_speech_recognizer(self):
        """Initialize speech recognition model."""
        try:
            # Use a thread to load the model (it's heavy)
            self.model_ready = threading.Event()
            self.whisper_model = None
            
            def load_model():
                try:
                    self.whisper_model = whisper.load_model("medium")
                    self.model_ready.set()
                    logger.info("Speech recognition model loaded")
                except Exception as e:
                    logger.error(f"Error loading speech model: {e}")
            
            threading.Thread(target=load_model, daemon=True).start()
        except Exception as e:
            logger.error(f"Error initializing speech recognizer: {e}")
    
    async def start(self):
        """Start voice interaction system."""
        logger.info("Starting voice interaction system...")
        
        if not self.porcupine:
            logger.error("Wake word detector not initialized. Voice interaction unavailable.")
            return
        
        if not self.model_ready.is_set():
            logger.info("Waiting for speech recognition model to load...")
            await asyncio.to_thread(self.model_ready.wait, timeout=60)
            if not self.model_ready.is_set():
                logger.error("Speech model failed to load. Voice interaction will use fallback.")
        
        # Reset state
        self._stop_event.clear()
        self._wake_word_detected.clear()
        
        # Start audio capture thread
        self._audio_capture_task = asyncio.create_task(self._capture_audio())
        
        # Start wake word detection
        self._wake_word_task = asyncio.create_task(self._detect_wake_word())
        
        # Start command processing
        self._command_task = asyncio.create_task(self._process_commands())
        
        logger.info("Voice interaction system started")
        self.is_listening = True
    
    async def stop(self):
        """Stop voice interaction system."""
        logger.info("Stopping voice interaction system...")
        
        # Signal stop
        self._stop_event.set()
        
        # Wait for tasks to complete
        tasks = []
        if hasattr(self, '_audio_capture_task'):
            tasks.append(self._audio_capture_task)
        if hasattr(self, '_wake_word_task'):
            tasks.append(self._wake_word_task)
        if hasattr(self, '_command_task'):
            tasks.append(self._command_task)
        
        if tasks:
            # Wait with timeout
            done, pending = await asyncio.wait(tasks, timeout=5)
            for task in pending:
                task.cancel()
        
        # Release resources
        self.is_listening = False
        if hasattr(self, 'porcupine') and self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
        
        logger.info("Voice interaction system stopped")
    
    async def _capture_audio(self):
        """Continuously capture audio from microphone."""
        try:
            def audio_callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Audio callback status: {status}")
                
                # Process audio data
                audio_data = indata.copy().flatten().astype(np.float32)
                
                # Add to audio queue
                try:
                    self._audio_queue.put_nowait(audio_data)
                except asyncio.QueueFull:
                    # If queue is full, remove oldest item
                    try:
                        self._audio_queue.get_nowait()
                        self._audio_queue.put_nowait(audio_data)
                    except (asyncio.QueueEmpty, asyncio.QueueFull):
                        pass
            
            # Start audio stream
            with sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.frame_length,
                dtype=np.float32
            ):
                logger.info("Audio capture started")
                
                # Run until stopped
                while not self._stop_event.is_set():
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Error in audio capture: {e}")
            if not self._stop_event.is_set():
                # Try to restart capture if it wasn't intentionally stopped
                await asyncio.sleep(1)
                asyncio.create_task(self._capture_audio())
    
    async def _detect_wake_word(self):
        """Process audio for wake word detection."""
        try:
            frame_length = self.porcupine.frame_length
            
            while not self._stop_event.is_set():
                # Get audio frame from queue
                try:
                    audio_frame = await asyncio.wait_for(self._audio_queue.get(), timeout=0.5)
                except (asyncio.TimeoutError, asyncio.QueueEmpty):
                    continue
                
                # Add to audio buffer for context
                self._audio_buffer = np.append(self._audio_buffer, audio_frame)
                
                # Keep only the recent buffer_seconds of audio
                max_samples = int(self.sample_rate * self.buffer_seconds)
                if len(self._audio_buffer) > max_samples:
                    self._audio_buffer = self._audio_buffer[-max_samples:]
                
                # Process audio in chunks matching what Porcupine expects
                while len(audio_frame) >= frame_length:
                    frame = audio_frame[:frame_length]
                    audio_frame = audio_frame[frame_length:]
                    
                    # Convert to correct format for Porcupine (int16)
                    pcm = (frame * 32767).astype(np.int16)
                    
                    if self.porcupine:
                        result = self.porcupine.process(pcm)
                        if result >= 0:  # Wake word detected
                            logger.info("Wake word detected!")
                            
                            # Clear the event in case it was already set
                            self._wake_word_detected.clear()
                            
                            # Set wake word event to trigger command processing
                            self._wake_word_detected.set()
                            
                            # Start a new recording with context
                            self._current_recording = self._audio_buffer.copy()
                            
                            # Wait for command processing to start
                            await asyncio.sleep(0.1)
                            break
                
        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
            if not self._stop_event.is_set():
                # Try to restart if it wasn't intentionally stopped
                await asyncio.sleep(1)
                asyncio.create_task(self._detect_wake_word())
    
    async def _record_command(self) -> Optional[np.ndarray]:
        """Record command after wake word until silence."""
        try:
            silence_samples = int(self.sample_rate * self.silence_seconds)
            silence_counter = 0
            recording_started = time.time()
            max_command_duration = 10  # Maximum 10 seconds for a command
            
            # Start with pre-wake audio context
            audio_data = self._current_recording.copy()
            
            # Record until silence or max duration
            while (time.time() - recording_started) < max_command_duration:
                if self._stop_event.is_set():
                    return None
                
                try:
                    # Get more audio
                    frame = await asyncio.wait_for(self._audio_queue.get(), timeout=0.5)
                    audio_data = np.append(audio_data, frame)
                    
                    # Check for silence
                    if np.max(np.abs(frame)) < self.silence_threshold:
                        silence_counter += len(frame)
                        if silence_counter >= silence_samples:
                            logger.debug("Silence detected, command recording complete")
                            break
                    else:
                        silence_counter = 0
                        
                except (asyncio.TimeoutError, asyncio.QueueEmpty):
                    continue
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error recording command: {e}")
            return None
    
    async def _transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio to text using Whisper."""
        try:
            if not self.whisper_model:
                logger.error("Speech model not loaded")
                return None
                
            # Save audio temporarily for processing
            temp_file = "temp_command.wav"
            wavfile.write(temp_file, self.sample_rate, audio_data)
            
            # Transcribe using Whisper (CPU-intensive)
            result = await asyncio.to_thread(
                self.whisper_model.transcribe, 
                temp_file,
                language="en"
            )
            
            # Clean up temporary file
            try:
                os.remove(temp_file)
            except:
                pass
                
            text = result["text"].strip()
            logger.info(f"Transcribed: {text}")
            return text
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None
    
    async def _process_commands(self):
        """Process voice commands after wake word detection."""
        try:
            while not self._stop_event.is_set():
                # Wait for wake word
                try:
                    await asyncio.wait_for(self._wake_word_detected.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Reset wake word event
                self._wake_word_detected.clear()
                
                # Visual/audio indication that we're listening
                self.is_processing = True
                
                # Record the command
                logger.info("Recording command...")
                audio_data = await self._record_command()
                if audio_data is None:
                    logger.warning("Command recording failed or interrupted")
                    self.is_processing = False
                    continue
                
                # Transcribe the command
                logger.info("Transcribing command...")
                command_text = await self._transcribe_audio(audio_data)
                if not command_text:
                    await self._speak("I apologize, but I couldn't understand that command.")
                    self.is_processing = False
                    continue
                
                # Process the command with the main system
                logger.info(f"Processing command: {command_text}")
                try:
                    system = await get_async_system_integrator()
                    
                    context = {
                        "source": "voice",
                        "user_id": "default",  # In real system, identify by voice
                        "timestamp": time.time()
                    }
                    
                    # Process command through main system
                    result = await system.process_input(command_text, context)
                    
                    # Speak the response
                    response_text = result.get("text", "I processed your request.")
                    await self._speak(response_text)
                    
                except Exception as e:
                    logger.error(f"Error processing command: {e}")
                    await self._speak("I apologize, but I encountered an error processing your command.")
                
                self.is_processing = False
                
        except Exception as e:
            logger.error(f"Error in command processor: {e}")
            if not self._stop_event.is_set():
                # Try to restart if it wasn't intentionally stopped
                await asyncio.sleep(1)
                asyncio.create_task(self._process_commands())
    
    async def _speak(self, text: str):
        """Generate and play speech response."""
        try:
            self.is_responding = True
            
            # Generate speech audio
            audio_data = await asyncio.to_thread(
                self.speech_synthesizer.synthesize,
                text
            )
            
            if audio_data is not None:
                # Play audio
                sd.play(audio_data, self.speech_synthesizer.sample_rate)
                sd.wait()
            else:
                logger.error("Speech synthesis failed")
            
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
        finally:
            self.is_responding = False
    
    def register_command_handler(self, command: str, handler: Callable):
        """
        Register a handler for a specific command.
        
        Args:
            command: Command trigger phrase
            handler: Callback function for the command
        """
        self._command_handlers[command.lower()] = handler
        self._commands.add(command.lower())
        logger.debug(f"Registered handler for command: {command}")
    
    @property
    def status(self) -> Dict[str, Any]:
        """Get current status of the voice system."""
        return {
            "listening": self.is_listening,
            "processing": self.is_processing,
            "responding": self.is_responding,
            "wake_word_active": self.porcupine is not None,
            "speech_model_loaded": self.model_ready.is_set() if hasattr(self, "model_ready") else False,
            "registered_commands": len(self._commands)
        }

# Create singleton instance
_voice_interaction_system = None

async def get_voice_interaction_system(config: Optional[Dict[str, Any]] = None):
    """Get singleton instance of VoiceInteractionSystem."""
    global _voice_interaction_system
    if _voice_interaction_system is None:
        _voice_interaction_system = VoiceInteractionSystem(config)
    return _voice_interaction_system