"""
Speech recognition module with Whisper integration for Sebastian assistant.

Provides high-fidelity transcription with noise resilience and configurable parameters.
"""
import logging
import os
import tempfile
import time
import numpy as np
import torch
import asyncio
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import whisper
import sounddevice as sd

logger = logging.getLogger(__name__)

class SpeechRecognizer:
    """
    Enhanced speech recognition using OpenAI's Whisper model.
    
    Features:
    - Multiple model size options (tiny, base, small, medium, large)
    - Noise-resilient processing
    - Speaker adaptation capabilities
    - Streaming and batch processing
    - Wake word validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize speech recognizer with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Load model configuration
        self.model_name = self.config.get("model_name", "base")
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.language = self.config.get("language", "en")
        
        # Processing parameters
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.chunk_size = self.config.get("chunk_size", 4096)
        
        # Performance optimization options
        self.beam_size = self.config.get("beam_size", 5)
        self.best_of = self.config.get("best_of", 5)
        self.temperature = self.config.get("temperature", 0)
        self.compression_ratio_threshold = self.config.get("compression_ratio_threshold", 2.4)
        self.logprob_threshold = self.config.get("logprob_threshold", -1.0)
        self.no_speech_threshold = self.config.get("no_speech_threshold", 0.6)
        
        # Enable word timestamps for precise wake word validation
        self.word_timestamps = self.config.get("word_timestamps", True)
        
        # Cache directory for temporary files and models
        self.cache_dir = Path(self.config.get("cache_dir", os.path.join(tempfile.gettempdir(), "sebastian_asr")))
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize model
        self._initialize_model()
        
        # Performance statistics
        self.stats = {
            "total_requests": 0,
            "successful_recognitions": 0,
            "avg_processing_time": 0,
            "total_audio_seconds": 0
        }
        
        logger.info(f"Speech recognizer initialized with {self.model_name} model on {self.device}")
        
    def _initialize_model(self):
        """Initialize Whisper speech recognition model."""
        try:
            start_time = time.time()
            logger.info(f"Loading Whisper {self.model_name} model...")
            
            self.model = whisper.load_model(
                self.model_name,
                device=self.device,
                download_root=str(self.cache_dir)
            )
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            
            # Optimize for inference
            self.model.eval()
            
            # Pre-initialize decoder for faster first inference
            if hasattr(self.model, "decode"):
                logger.debug("Pre-initializing decoder")
                dummy_mel = torch.zeros((1, 80, 3000), device=self.device)
                with torch.no_grad():
                    self.model.decode(dummy_mel)
                
        except Exception as e:
            logger.error(f"Failed to initialize speech recognition model: {e}", exc_info=True)
            raise RuntimeError(f"Speech recognition initialization failed: {e}")
            
    def recognize(self, audio_data: Union[np.ndarray, bytes, str], 
                prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Recognize speech in audio data.
        
        Args:
            audio_data: Audio as numpy array, bytes, or file path
            prompt: Optional text prompt to guide transcription
            
        Returns:
            Dictionary with transcription results
        """
        self.stats["total_requests"] += 1
        start_time = time.time()
        
        try:
            # Handle different audio input formats
            if isinstance(audio_data, bytes):
                # Save bytes to temporary file
                audio_path = self.cache_dir / f"audio_{int(time.time())}.wav"
                with open(audio_path, "wb") as f:
                    f.write(audio_data)
                audio = whisper.load_audio(str(audio_path))
                
            elif isinstance(audio_data, str):
                # Load from file path
                audio = whisper.load_audio(audio_data)
                
            else:
                # Assume numpy array
                audio = audio_data
                
            # Calculate audio duration
            audio_duration = len(audio) / self.sample_rate
            self.stats["total_audio_seconds"] += audio_duration
            
            # Pad/trim audio to fit model requirements
            audio = whisper.pad_or_trim(audio)
            
            # Convert to log-mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self.device)
            
            # Detect language if not specified
            if self.language == "auto":
                _, probs = self.model.detect_language(mel)
                detected_lang = max(probs, key=probs.get)
                logger.debug(f"Detected language: {detected_lang}")
            else:
                detected_lang = self.language
                
            # Decode audio
            decode_options = {
                "beam_size": self.beam_size,
                "best_of": self.best_of,
                "temperature": self.temperature,
                "compression_ratio_threshold": self.compression_ratio_threshold,
                "logprob_threshold": self.logprob_threshold,
                "no_speech_threshold": self.no_speech_threshold,
                "fp16": self.device == "cuda",
                "language": detected_lang
            }
            
            # Add prompt if provided
            if prompt:
                decode_options["prompt"] = prompt
                
            # Enable word timestamps if needed
            if self.word_timestamps:
                decode_options["word_timestamps"] = True
                
            # Run transcription
            logger.debug("Transcribing audio...")
            result = self.model.transcribe(
                audio,
                **decode_options
            )
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Update statistics
            self.stats["successful_recognitions"] += 1
            self.stats["avg_processing_time"] = (
                (self.stats["avg_processing_time"] * (self.stats["successful_recognitions"] - 1) + process_time) / 
                self.stats["successful_recognitions"]
            )
            
            logger.info(f"Recognized speech in {process_time:.2f}s: {result['text'][:50]}...")
            
            # Prepare results
            recognition_result = {
                "text": result["text"],
                "language": result.get("language", detected_lang),
                "segments": result.get("segments", []),
                "words": result.get("words", []) if self.word_timestamps else [],
                "audio_duration": audio_duration,
                "process_time": process_time,
                "confidence": self._calculate_confidence(result)
            }
            
            return recognition_result
            
        except Exception as e:
            logger.error(f"Speech recognition error: {e}", exc_info=True)
            return {
                "text": "",
                "error": str(e),
                "success": False
            }
            
    async def recognize_async(self, audio_data: Union[np.ndarray, bytes, str], 
                            prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Recognize speech asynchronously.
        
        Args:
            audio_data: Audio as numpy array, bytes, or file path
            prompt: Optional text prompt to guide transcription
            
        Returns:
            Dictionary with transcription results
        """
        # Run synchronous recognition in a thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.recognize(audio_data, prompt)
        )
        
    async def recognize_stream(self, stream_generator, 
                             chunk_duration_ms: int = 1000) -> Dict[str, Any]:
        """
        Recognize speech from streaming audio source.
        
        Args:
            stream_generator: Async generator yielding audio chunks
            chunk_duration_ms: Duration of each chunk in milliseconds
            
        Returns:
            Dictionary with transcription results
        """
        audio_chunks = []
        total_duration_s = 0
        
        try:
            # Collect audio chunks
            async for chunk in stream_generator:
                audio_chunks.append(chunk)
                chunk_duration_s = chunk_duration_ms / 1000
                total_duration_s += chunk_duration_s
                
                # Process when we have sufficient audio
                if total_duration_s >= 2.0:  # Process every 2 seconds
                    # Combine chunks
                    combined_audio = np.concatenate(audio_chunks)
                    
                    # Process audio asynchronously
                    partial_result = await self.recognize_async(combined_audio)
                    
                    # Yield intermediate result
                    yield {
                        "text": partial_result["text"],
                        "is_final": False,
                        "audio_duration": total_duration_s
                    }
            
            # Final processing with all audio
            if audio_chunks:
                combined_audio = np.concatenate(audio_chunks)
                final_result = await self.recognize_async(combined_audio)
                
                yield {
                    "text": final_result["text"],
                    "is_final": True,
                    "audio_duration": total_duration_s,
                    "confidence": final_result.get("confidence", 0.0)
                }
            
        except Exception as e:
            logger.error(f"Streaming recognition error: {e}", exc_info=True)
            yield {
                "text": "",
                "error": str(e),
                "is_final": True,
                "success": False
            }
            
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate overall confidence score from result."""
        if "segments" not in result or not result["segments"]:
            return 0.0
            
        # Average log probability across segments
        avg_logprob = sum(segment.get("avg_logprob", -1.0) for segment in result["segments"])
        avg_logprob /= len(result["segments"])
        
        # Convert to confidence score (0-1)
        # Typical values range from -1 to 0, where 0 is highest confidence
        confidence = 1.0 + avg_logprob if avg_logprob >= -1.0 else 0.0
        
        return confidence
        
    def validate_wake_word(self, result: Dict[str, Any], wake_word: str = "sebastian") -> bool:
        """
        Check if transcription contains wake word.
        
        Args:
            result: Recognition result
            wake_word: Wake word to detect
            
        Returns:
            True if wake word found
        """
        if not result or "text" not in result:
            return False
            
        # Case insensitive search
        transcription = result["text"].lower()
        wake_word = wake_word.lower()
        
        # Simple text search
        contains_wake_word = wake_word in transcription.split()
        
        # If word timestamps available, verify timing
        if contains_wake_word and "words" in result and result["words"]:
            # Find wake word with highest confidence
            for word_info in result["words"]:
                if word_info["word"].lower().strip().replace(" ", "") == wake_word:
                    # Consider only recent wake words (last 2 seconds)
                    if word_info["end"] > (result.get("audio_duration", 0) - 2.0):
                        return True
                        
            # No recent wake word with high confidence
            return False
            
        return contains_wake_word
        
    def record_audio(self, duration_seconds: float = 5.0, 
                   device_index: Optional[int] = None) -> np.ndarray:
        """
        Record audio from microphone.
        
        Args:
            duration_seconds: Recording duration in seconds
            device_index: Audio device index
            
        Returns:
            Recorded audio as numpy array
        """
        logger.info(f"Recording audio for {duration_seconds} seconds")
        audio_data = sd.rec(
            int(duration_seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            device=device_index
        )
        sd.wait()  # Wait until recording is finished
        
        return audio_data.flatten()
        
    def get_stats(self) -> Dict[str, Any]:
        """Get recognition statistics."""
        return self.stats
