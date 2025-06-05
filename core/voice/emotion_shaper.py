"""
Emotion shaping for Sebastian's synthesized voice.

Adds subtle emotional qualities to speech synthesis while maintaining
the composed, elegant vocal character of Sebastian Michaelis.
"""
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class EmotionShaper:
    """
    Applies emotional qualities to Sebastian's synthesized speech.
    
    Features:
    - Subtle emotional variations appropriate for Sebastian's character
    - Adjustments to pitch, speed, energy, and articulation
    - Emotion intensity controls with appropriate restraint
    - Context-aware emotion application
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize emotion shaper with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Default emotional restraint (0-1, higher means more restraint)
        self.emotional_restraint = self.config.get("emotional_restraint", 0.8)
        
        # Emotion profiles
        self._initialize_emotion_profiles()
        
        logger.info("Voice emotion shaper initialized")
        
    def _initialize_emotion_profiles(self):
        """Initialize emotion parameter profiles."""
        # Emotion parameter ranges are defined as (base_value, min_value, max_value)
        # Parameters maintained within narrow ranges to ensure Sebastian's voice remains recognizable
        
        # Base voice profile
        self.base_profile = {
            "pitch_factor": 1.0,      # Default pitch
            "speed_factor": 1.0,      # Default speed
            "energy_factor": 1.0,     # Default energy
            "breathiness": 0.05,      # Very slight breathiness
            "precision": 0.9,         # High articulation precision
            "formality": 0.95         # Very formal articulation
        }
        
        # Emotion-specific profile adjustments (always subtle for Sebastian)
        self.emotion_profiles = {
            # Positive emotions - always understated
            "pleased": {
                "pitch_factor": (1.03, 1.0, 1.05),    # Slightly higher pitch
                "speed_factor": (1.02, 1.0, 1.05),    # Slightly faster
                "energy_factor": (1.05, 1.0, 1.1),    # Slightly more energy
                "breathiness": (0.03, 0.02, 0.05),    # Less breathy
                "precision": (0.92, 0.9, 0.95),       # More precise
                "formality": (0.97, 0.95, 0.98)       # More formal
            },
            "satisfied": {
                "pitch_factor": (1.02, 1.0, 1.03),    # Barely higher pitch
                "speed_factor": (1.0, 0.98, 1.02),    # Normal speed
                "energy_factor": (1.02, 1.0, 1.05),   # Slightly more energy
                "breathiness": (0.04, 0.03, 0.05),    # Normal breathiness
                "precision": (0.93, 0.9, 0.95),       # More precise
                "formality": (0.96, 0.95, 0.98)       # More formal
            },
            "amused": {
                "pitch_factor": (1.04, 1.02, 1.06),   # Higher pitch
                "speed_factor": (1.03, 1.0, 1.05),    # Slightly faster
                "energy_factor": (1.03, 1.0, 1.07),   # Slightly more energy
                "breathiness": (0.04, 0.03, 0.06),    # Normal breathiness
                "precision": (0.92, 0.9, 0.94),       # More precise
                "formality": (0.93, 0.91, 0.95)       # Slightly less formal
            },
            
            # Negative emotions - extremely subtle for Sebastian
            "displeased": {
                "pitch_factor": (0.98, 0.97, 1.0),    # Slightly lower pitch
                "speed_factor": (0.98, 0.95, 1.0),    # Slightly slower
                "energy_factor": (0.97, 0.95, 1.0),   # Slightly less energy
                "breathiness": (0.06, 0.05, 0.08),    # Slightly more breathy
                "precision": (0.95, 0.93, 0.97),      # More precise articulation
                "formality": (0.98, 0.97, 0.99)       # More formal
            },
            "concerned": {
                "pitch_factor": (0.97, 0.95, 0.99),   # Lower pitch
                "speed_factor": (0.95, 0.93, 0.98),   # Slower
                "energy_factor": (0.96, 0.94, 0.99),  # Less energy
                "breathiness": (0.07, 0.05, 0.09),    # More breathy
                "precision": (0.96, 0.94, 0.98),      # Very precise articulation
                "formality": (0.98, 0.97, 0.99)       # More formal
            },
            "irritated": {
                "pitch_factor": (0.99, 0.97, 1.01),   # Barely altered pitch
                "speed_factor": (1.02, 1.0, 1.05),    # Slightly faster
                "energy_factor": (1.03, 1.0, 1.07),   # More energy
                "breathiness": (0.03, 0.02, 0.04),    # Less breathy (more clipped)
                "precision": (0.97, 0.95, 0.99),      # Extremely precise
                "formality": (0.99, 0.98, 1.0)        # Extremely formal
            },
            
            # Stronger emotions - rare for Sebastian
            "angry": {
                "pitch_factor": (0.96, 0.94, 0.98),   # Lower pitch
                "speed_factor": (1.03, 1.0, 1.06),    # Faster
                "energy_factor": (1.08, 1.05, 1.12),  # More energy
                "breathiness": (0.02, 0.01, 0.03),    # Very little breathiness
                "precision": (0.98, 0.96, 1.0),       # Extremely precise
                "formality": (0.96, 0.94, 0.98)       # Slightly less formal (intensity)
            },
            "protective": {
                "pitch_factor": (0.95, 0.93, 0.97),   # Lower pitch
                "speed_factor": (1.05, 1.03, 1.08),   # Faster
                "energy_factor": (1.1, 1.05, 1.15),   # More energy
                "breathiness": (0.02, 0.01, 0.03),    # Very little breathiness
                "precision": (0.98, 0.96, 1.0),       # Extremely precise
                "formality": (0.95, 0.94, 0.97)       # Slightly less formal (urgency)
            },
            
            # Special modes for Sebastian
            "demonic": {
                "pitch_factor": (0.93, 0.9, 0.95),    # Much lower pitch
                "speed_factor": (0.95, 0.93, 0.97),   # Slower
                "energy_factor": (1.05, 1.0, 1.1),    # More energy
                "breathiness": (0.02, 0.01, 0.03),    # Almost no breathiness
                "precision": (0.99, 0.98, 1.0),       # Perfect precision
                "formality": (0.97, 0.95, 0.99)       # Extremely formal
            },
            "playful": {
                "pitch_factor": (1.05, 1.03, 1.07),   # Higher pitch
                "speed_factor": (1.04, 1.02, 1.06),   # Faster
                "energy_factor": (1.04, 1.02, 1.06),  # More energy
                "breathiness": (0.04, 0.03, 0.05),    # Normal breathiness
                "precision": (0.9, 0.88, 0.92),       # Less precise
                "formality": (0.92, 0.9, 0.94)        # Less formal
            }
        }
        
    def shape_emotion(self, emotion: str, intensity: float = 0.5, 
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Generate emotion parameters for speech synthesis.
        
        Args:
            emotion: Emotion to express
            intensity: Emotion intensity (0-1)
            context: Optional context information
            
        Returns:
            Dictionary of emotion parameters
        """
        # Apply emotional restraint to intensity
        restrained_intensity = intensity * (1 - self.emotional_restraint)
        
        # Default to base profile
        params = self.base_profile.copy()
        
        # If emotion not recognized or intensity too low after restraint, use base profile
        if emotion not in self.emotion_profiles or restrained_intensity < 0.1:
            return params
            
        # Get emotion profile
        profile = self.emotion_profiles[emotion]
        
        # Apply emotion profile parameters with intensity scaling
        for param, (base_value, min_value, max_value) in profile.items():
            # Calculate parameter value based on restraint and intensity
            if intensity > 0.5:
                # High intensity moves toward max value
                value_range = max_value - base_value
                param_value = base_value + (value_range * (restrained_intensity - 0.5) * 2)
            else:
                # Low intensity moves toward min value
                value_range = base_value - min_value
                param_value = base_value - (value_range * (0.5 - restrained_intensity) * 2)
                
            params[param] = param_value
            
        # Adjust for context if provided
        if context:
            params = self._adjust_for_context(params, context, emotion)
            
        # Apply final moderation to ensure Sebastian's voice remains recognizable
        params = self._moderate_parameters(params)
            
        return params
        
    def _adjust_for_context(self, params: Dict[str, float], context: Dict[str, Any], 
                           emotion: str) -> Dict[str, float]:
        """Adjust emotion parameters based on context."""
        # Adjust formality based on addressee
        relationship = context.get("relationship", "master")
        
        if relationship == "master":
            # Always highly formal with master
            params["formality"] = max(params["formality"], 0.95)
            
        elif relationship == "enemy":
            # Allow slightly more emotional expression with enemies
            for param in ["pitch_factor", "speed_factor", "energy_factor"]:
                # Amplify deviation from 1.0
                deviation = params[param] - 1.0
                params[param] = 1.0 + (deviation * 1.2)  # 20% more expressive
                
        # Adjust for urgency
        if context.get("urgent", False):
            params["speed_factor"] *= 1.05  # 5% faster
            params["energy_factor"] *= 1.05  # 5% more energy
            
        # Adjust for public setting
        if context.get("public", False):
            params["precision"] = min(params["precision"] + 0.02, 0.99)  # More precise
            params["formality"] = min(params["formality"] + 0.02, 0.99)  # More formal
            
        return params
        
    def _moderate_parameters(self, params: Dict[str, float]) -> Dict[str, float]:
        """Apply final moderation to ensure voice remains recognizable."""
        # Hard limits to ensure voice doesn't change too drastically
        limits = {
            "pitch_factor": (0.9, 1.1),      # Don't alter pitch too much
            "speed_factor": (0.9, 1.1),      # Don't alter speed too much
            "energy_factor": (0.9, 1.2),     # Allow slightly more energy variation
            "breathiness": (0.01, 0.1),      # Keep breathiness in narrow range
            "precision": (0.85, 0.99),       # Always fairly precise
            "formality": (0.9, 0.99)         # Always fairly formal
        }
        
        # Apply limits
        for param, (min_limit, max_limit) in limits.items():
            if param in params:
                params[param] = max(min_limit, min(params[param], max_limit))
                
        return params
        
    def get_tts_params(self, emotion: str, intensity: float = 0.5,
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Get parameters specifically for TTS engine.
        
        Args:
            emotion: Emotion to express
            intensity: Emotion intensity (0-1)
            context: Optional context information
            
        Returns:
            Dictionary of TTS-compatible parameters
        """
        # Get full emotion parameters
        full_params = self.shape_emotion(emotion, intensity, context)
        
        # Extract only the parameters needed for TTS
        tts_params = {
            "pitch_factor": full_params["pitch_factor"],
            "speed_factor": full_params["speed_factor"],
            "energy_factor": full_params["energy_factor"]
        }
        
        return tts_params
        
    def adapt_ssml(self, text: str, emotion: str, intensity: float = 0.5) -> str:
        """
        Adapt text with SSML tags for emotional expression.
        
        Args:
            text: Input text
            emotion: Emotion to express
            intensity: Emotion intensity
            
        Returns:
            Text with SSML emotional markup
        """
        # Apply emotional restraint to intensity
        restrained_intensity = intensity * (1 - self.emotional_restraint)
        
        # Skip SSML for low intensity or unrecognized emotions
        if emotion not in self.emotion_profiles or restrained_intensity < 0.2:
            return text
            
        result = text
        
        # Add prosody tags based on emotion
        if emotion in ["concerned", "protective"]:
            result = f"<prosody rate='{int(-5 * restrained_intensity)}%' pitch='{int(-3 * restrained_intensity)}%'>{result}</prosody>"
            
        elif emotion in ["pleased", "satisfied"]:
            result = f"<prosody rate='{int(2 * restrained_intensity)}%' pitch='{int(2 * restrained_intensity)}%'>{result}</prosody>"
            
        elif emotion == "angry":
            result = f"<prosody rate='{int(5 * restrained_intensity)}%' volume='{int(10 * restrained_intensity)}%'>{result}</prosody>"
            
        elif emotion == "demonic":
            result = f"<prosody pitch='{int(-10 * restrained_intensity)}%' volume='{int(5 * restrained_intensity)}%'>{result}</prosody>"
            
        # Add specific emotional emphasis for certain phrases
        if "my lord" in result and emotion in ["protective", "concerned"]:
            result = result.replace("my lord", "<emphasis level='moderate'>my lord</emphasis>")
            
        if "intruder" in result and emotion in ["angry", "protective"]:
            result = result.replace("intruder", "<emphasis level='strong'>intruder</emphasis>")
            
        return result

