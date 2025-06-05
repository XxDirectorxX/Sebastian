"""
Voice command processing loop for Sebastian assistant.

Implements continuous listening with wake word detection and voice command processing.
"""
import asyncio
import logging
import sounddevice as sd
import numpy as np
import queue
import threading
from datetime import datetime
import time
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

class VoiceCommandLoop:
    """
    Implements continuous voice command processing with wake word detection,
    voice recognition, and command dispatching.
    """
    
    def __init__(self, 
                system_integrator,
                wake_word: str = "sebastian",
                security_manager = None,
                sample_rate: int = 16000,
                device_index: Optional[int] = None):
        """
        Initialize voice command processing loop.
        
        Args:
            system_integrator: Reference to system integrator
            wake_word: Wake word to trigger assistant
            security_manager: Reference to security manager for voice authentication
            sample_rate: Audio sample rate
            device_index: Audio device index to use
        """
        self.system_integrator = system_integrator
        self.wake_word = wake_word
        self.security_manager = security_manager
        self.sample_rate = sample_rate
        self.device_index = device_index
        
        # Audio buffer
        self.audio_queue = queue.Queue()
        self.buffer_size = 1024
        self.frame_duration_ms = 1000 * self.buffer_size // self.sample_rate
        
        # State
        self.running = False
        self.listening_for_command = False
        self.recognized_user = None
        
        # Import components
        try:
            from core.voice.speech_recognition import SpeechRecognizer
            import pvporcupine
        except ImportError as e:
            logger.error(f"Failed to import voice components: {e}")
            raise
            
        # Initialize wake word detector
        try:
            self.porcupine = pvporcupine.create(keywords=[wake_word])
            logger.info(f"Wake word detector initialized with wake word: {wake_word}")
        except Exception as e:
            logger.error(f"Failed to initialize wake word detector: {e}")
            self.porcupine = None
            
        # Initialize speech recognizer
        try:
            self.recognizer = SpeechRecognizer()
            logger.info("Speech recognizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize speech recognizer: {e}")
            self.recognizer = None
            
        logger.info("Voice command loop initialized")
        
    async def run(self):
        """Run the voice command loop asynchronously."""
        if not self.porcupine or not self.recognizer:
            logger.error("Cannot run voice command loop: Components not properly initialized")
            return
            
        self.running = True
        
        # Start audio recording in a separate thread
        threading.Thread(target=self._audio_callback_thread, daemon=True).start()
        
        try:
            logger.info("Voice command loop started")
            while self.running:
                await self._process_audio()
                await asyncio.sleep(0.1)  # Small sleep to avoid busy loop
        except Exception as e:
            logger.error(f"Error in voice command loop: {e}", exc_info=True)
        finally:
            self._cleanup()
            
    def _audio_callback_thread(self):
        """Audio recording callback thread."""
        try:
            with sd.InputStream(
                callback=self._audio_callback,
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.buffer_size,
                device=self.device_index
            ):
                logger.info("Started audio stream")
                while self.running:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in audio callback thread: {e}", exc_info=True)
            self.running = False
            
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream."""
        if status:
            logger.warning(f"Audio callback status: {status}")
            
        # Add audio data to queue
        self.audio_queue.put(indata.copy())
            
    async def _process_audio(self):
        """Process audio from the queue."""
        try:
            # Get audio data from queue
            if self.audio_queue.empty():
                return
                
            audio_data = self.audio_queue.get()
            
            # Process wake word if not already listening for command
            if not self.listening_for_command:
                # Convert to the format expected by Porcupine
                pcm = audio_data.flatten().astype(np.int16)
                
                # Check for wake word
                result = self.porcupine.process(pcm)
                if result >= 0:
                    logger.info("Wake word detected!")
                    await self._handle_wake_word()
            else:
                # Add to command buffer
                await self._process_command_audio(audio_data)
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            
    async def _handle_wake_word(self):
        """Handle wake word detection."""
        # Clear any existing audio
        while not self.audio_queue.empty():
            self.audio_queue.get()
            
        # Play acknowledgment sound or speak acknowledgment
        # For now, just log it
        logger.info("Listening for command...")
        
        # Set state to listen for command
        self.listening_for_command = True
        self.command_buffer = np.array([], dtype=np.float32)
        self.command_start_time = time.time()
        
    async def _process_command_audio(self, audio_data):
        """Process audio for command recognition."""
        # Add to command buffer
        self.command_buffer = np.append(self.command_buffer, audio_data.flatten())
        
        # Check if we have enough audio or too much time has passed
        current_time = time.time()
        elapsed_time = current_time - self.command_start_time
        
        # If 5 seconds of audio or 0.5 seconds of silence
        if elapsed_time >= 5.0 or self._detect_silence(audio_data, threshold=0.01):
            # Process the command
            await self._recognize_and_process_command()
            
            # Reset listening state
            self.listening_for_command = False
            
    def _detect_silence(self, audio_data, threshold=0.01):
        """Detect if audio contains silence."""
        # Skip if we don't have enough audio yet
        if len(self.command_buffer) < self.sample_rate:
            return False
            
        # Check last second of audio for silence
        last_second = audio_data[-self.sample_rate:]
        rms = np.sqrt(np.mean(np.square(last_second)))
        
        return rms < threshold
            
    async def _recognize_and_process_command(self):
        """Recognize speech in command buffer and process it."""
        try:
            # Convert audio to format needed by recognizer
            audio_data = self.command_buffer.astype(np.float32)
            
            # Recognize speech
            logger.info("Processing command audio...")
            text = await self._recognize_speech(audio_data)
            
            if not text:
                logger.info("No speech recognized")
                return
                
            logger.info(f"Recognized command: {text}")
            
            # Identify user if security is enabled
            user_id = "voice_user"
            if self.security_manager:
                # This would use voice biometrics to identify the user
                # For now, just use a default user
                pass
                
            # Process command
            context = {
                "user_id": user_id,
                "interaction_mode": "voice",
                "timestamp": datetime.now().isoformat()
            }
            
            # Process command through system integrator
            result = await self.system_integrator.process_input_async(text, context)
            
            # Speak response
            await self._speak_response(result["text"])
            
        except Exception as e:
            logger.error(f"Error processing command: {e}", exc_info=True)
            await self._speak_response("I apologize, but I encountered an error processing your command.")
            
    async def _recognize_speech(self, audio_data):
        """Recognize speech in audio data."""
        if hasattr(self.recognizer, "recognize_async"):
            # Use async version if available
            return await self.recognizer.recognize_async(audio_data, self.sample_rate)
        else:
            # Fallback to synchronous version
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.recognizer.recognize(audio_data, self.sample_rate)
            )
            
    async def _speak_response(self, text):
        """Speak response text."""
        try:
            # This would use text-to-speech to speak the response
            # For now, just log it
            logger.info(f"Speaking response: {text}")
            
            # Here you would call the TTS engine
            # For example:
            # from core.voice.speech_synthesis import SpeechSynthesizer
            # synthesizer = SpeechSynthesizer()
            # audio_data = await synthesizer.synthesize_speech(text)
            # play_audio(audio_data)
            
        except Exception as e:
            logger.error(f"Error speaking response: {e}", exc_info=True)
            
    def _cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up voice command loop resources")
        
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
            
        logger.info("Voice command loop stopped")
            
    def stop(self):
        """Stop the voice command loop."""
        logger.info("Stopping voice command loop")
        self.running = False

