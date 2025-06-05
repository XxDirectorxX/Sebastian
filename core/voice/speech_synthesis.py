"""
Speech synthesis module for Sebastian's distinctive voice.

Implements high-quality text-to-speech with formal British accent and
sophisticated articulation patterns.
"""
import logging
import os
import time
import numpy as np
import torch
import soundfile as sf
import asyncio
from typing import Optional, Dict, Any, Union, List, BinaryIO
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)

class SpeechSynthesizer:
    """
    Speech synthesis for Sebastian's formal British voice.
    
    Features:
    - High-quality neural TTS for Sebastian's voice
    - Formal British accent with precise diction
    - Adjustable speaking style and emotion parameters
    - Support for streaming and file output
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize speech synthesizer with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Voice model configuration
        self.voice_model_path = Path(self.config.get("voice_model", "assets/voice_models/sebastian_voice_model.pth"))
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = self.config.get("sample_rate", 22050)
        
        # Voice characteristics
        self.base_pitch = self.config.get("pitch", 0.85)  # Lower pitch (masculine, authoritative)
        self.base_speed = self.config.get("speed", 0.95)  # Slightly slower for formality
        self.base_energy = self.config.get("energy", 1.0)  # Regular energy
        
        # British pronunciation dictionary
        self.british_dict_path = self.config.get("british_dict", "assets/voice_models/british_pronunciation.json")
        
        # Cache directory for output files
        self.cache_dir = Path(self.config.get("cache_dir", os.path.join(tempfile.gettempdir(), "sebastian_tts")))
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize TTS model
        self._initialize_model()
        
        logger.info("Speech synthesizer initialized")
        
    def _initialize_model(self):
        """Initialize text-to-speech model."""
        try:
            start_time = time.time()
            logger.info("Loading TTS model...")
            
            # Verify model file exists
            if not self.voice_model_path.exists():
                raise FileNotFoundError(f"Voice model not found at {self.voice_model_path}")
            
            # Import TTS library only when needed to avoid dependencies for non-voice systems
            try:
                import TTS
                from TTS.utils.synthesizer import Synthesizer
            except ImportError:
                logger.error("TTS library not found. Please install it with: pip install TTS")
                raise
                
            # Load appropriate TTS model - model type depends on what's available
            # This implementation assumes a pre-trained model specifically for Sebastian's voice
            if "tacotron" in str(self.voice_model_path).lower():
                # Tacotron2 + HiFiGAN vocoder
                self.synthesizer = Synthesizer(
                    tts_checkpoint=str(self.voice_model_path),
                    tts_config_path=str(self.voice_model_path).replace(".pth", ".json"),
                    vocoder_checkpoint=self.config.get("vocoder_path", ""),
                    vocoder_config=self.config.get("vocoder_config", ""),
                    use_cuda=self.device == "cuda"
                )
            elif "vits" in str(self.voice_model_path).lower():
                # VITS model (preferred)
                self.synthesizer = Synthesizer(
                    tts_checkpoint=str(self.voice_model_path),
                    tts_config_path=str(self.voice_model_path).replace(".pth", ".json"),
                    use_cuda=self.device == "cuda"
                )
            else:
                # Default to YourTTS for voice cloning capability
                self.synthesizer = Synthesizer(
                    tts_checkpoint=str(self.voice_model_path),
                    tts_config_path=str(self.voice_model_path).replace(".pth", ".json"),
                    use_cuda=self.device == "cuda"
                )
                
            # Load British pronunciation dictionary if available
            self.british_dict = {}
            if os.path.exists(self.british_dict_path):
                try:
                    import json
                    with open(self.british_dict_path, 'r') as f:
                        self.british_dict = json.load(f)
                    logger.info("Loaded British pronunciation dictionary")
                except Exception as e:
                    logger.warning(f"Could not load British pronunciation dictionary: {e}")
            
            load_time = time.time() - start_time
            logger.info(f"TTS model loaded in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS model: {e}", exc_info=True)
            # Create a dummy synthesizer for graceful degradation
            self.synthesizer = None
            raise RuntimeError(f"Speech synthesis initialization failed: {e}")
            
    def synthesize(self, text: str, emotion_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            emotion_params: Optional emotion parameters to apply
            
        Returns:
            Dictionary with synthesis results
        """
        if not self.synthesizer:
            logger.error("TTS model not initialized")
            return {"error": "TTS model not initialized", "success": False}
            
        try:
            start_time = time.time()
            
            # Apply British pronunciation adjustments
            text = self._apply_british_pronunciation(text)
            
            # Prepare voice parameters
            params = self._prepare_voice_params(emotion_params)
            
            # Synthesize speech
            logger.debug(f"Synthesizing text: {text[:50]}...")
            
            # Apply sentence-level SSML-like processing for better prosody
            processed_text = self._process_text_for_synthesis(text)
            
            # Call synthesizer with parameters
            # Implementation depends on the specific TTS model type
            if hasattr(self.synthesizer, 'tts_with_params'):
                wav = self.synthesizer.tts_with_params(
                    processed_text, 
                    speed=params["speed"],
                    energy=params["energy"],
                    pitch=params["pitch"]
                )
            else:
                # Fallback for models without parameter support
                wav = self.synthesizer.tts(processed_text)
                
            # Calculate processing time
            process_time = time.time() - start_time
            
            logger.info(f"Synthesized speech in {process_time:.2f}s")
            
            return {
                "audio": wav,
                "sample_rate": self.sample_rate,
                "text": text,
                "process_time": process_time,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Speech synthesis error: {e}", exc_info=True)
            return {
                "error": str(e),
                "success": False
            }
            
    def _apply_british_pronunciation(self, text: str) -> str:
        """Apply British pronunciation patterns to text."""
        # Replace American spellings with British equivalents
        result = text
        
        # Apply dictionary-based replacements
        for american, british in self.british_dict.items():
            result = result.replace(american, british)
            
        # Apply common pronunciation patterns
        result = result.replace("r ", "r ")  # Non-rhotic accent hint
        
        return result
        
    def _prepare_voice_params(self, emotion_params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Prepare voice parameters with emotion modifications."""
        # Start with base parameters
        params = {
            "pitch": self.base_pitch,
            "speed": self.base_speed,
            "energy": self.base_energy
        }
        
        # Apply emotion parameters if provided
        if emotion_params:
            # Adjust pitch (0.7-1.3)
            params["pitch"] *= emotion_params.get("pitch_factor", 1.0)
            params["pitch"] = max(0.7, min(1.3, params["pitch"]))
            
            # Adjust speed (0.7-1.3)
            params["speed"] *= emotion_params.get("speed_factor", 1.0)
            params["speed"] = max(0.7, min(1.3, params["speed"]))
            
            # Adjust energy (0.5-1.5)
            params["energy"] *= emotion_params.get("energy_factor", 1.0)
            params["energy"] = max(0.5, min(1.5, params["energy"]))
            
        return params
        
    def _process_text_for_synthesis(self, text: str) -> str:
        """Process text for better synthesis with sentence-level analysis."""
        # Insert strategic pauses for more formal, measured speech
        result = text
        
        # Add pauses after periods, question marks, exclamation points
        result = result.replace(". ", ". <break time='300ms'/> ")
        result = result.replace("? ", "? <break time='300ms'/> ")
        result = result.replace("! ", "! <break time='300ms'/> ")
        
        # Add slight pauses for commas and semicolons
        result = result.replace(", ", ", <break time='150ms'/> ")
        result = result.replace("; ", "; <break time='200ms'/> ")
        
        # Add emphasis to "I am simply one hell of a butler"
        if "one hell of a butler" in result.lower():
            result = result.replace(
                "one hell of a butler", 
                "<emphasis level='strong'>one hell of a butler</emphasis>"
            )
            
        # Add subtle emphasis for "Yes, my lord" and variants
        for phrase in ["Yes, my lord", "Yes, young master", "As you wish"]:
            if phrase in result:
                result = result.replace(
                    phrase,
                    f"<emphasis level='moderate'>{phrase}</emphasis>"
                )
                
        return result
        
    async def synthesize_async(self, text: str, 
                             emotion_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Synthesize speech asynchronously.
        
        Args:
            text: Text to synthesize
            emotion_params: Optional emotion parameters
            
        Returns:
            Dictionary with synthesis results
        """
        # Run synchronous synthesis in a thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.synthesize(text, emotion_params)
        )
        
    def save_to_file(self, synthesis_result: Dict[str, Any], file_path: str) -> bool:
        """
        Save synthesized speech to file.
        
        Args:
            synthesis_result: Result from synthesize() method
            file_path: Output file path
            
        Returns:
            True if successful
        """
        if not synthesis_result.get("success", False):
            logger.error("Cannot save unsuccessful synthesis result")
            return False
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Write audio file
            sf.write(
                file_path,
                synthesis_result["audio"],
                synthesis_result["sample_rate"]
            )
            
            logger.info(f"Saved synthesized speech to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save audio file: {e}", exc_info=True)
            return False
            
    async def save_to_file_async(self, synthesis_result: Dict[str, Any], file_path: str) -> bool:
        """
        Save synthesized speech to file asynchronously.
        
        Args:
            synthesis_result: Result from synthesize() method
            file_path: Output file path
            
        Returns:
            True if successful
        """
        # Run synchronous file save in a thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.save_to_file(synthesis_result, file_path)
        )
        
    def play_audio(self, synthesis_result: Dict[str, Any]) -> bool:
        """
        Play synthesized speech through speakers.
        
        Args:
            synthesis_result: Result from synthesize() method
            
        Returns:
            True if successful
        """
        if not synthesis_result.get("success", False):
            logger.error("Cannot play unsuccessful synthesis result")
            return False
            
        try:
            # Import sounddevice for audio playback
            import sounddevice as sd
            
            # Play audio
            sd.play(
                synthesis_result["audio"],
                synthesis_result["sample_rate"]
            )
            sd.wait()  # Wait until audio is done playing
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to play audio: {e}", exc_info=True)
            return False
