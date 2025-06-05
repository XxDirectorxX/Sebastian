"""
Dialogue management system for Sebastian personality simulation.

Orchestrates the application of tone, mannerisms, and emotions to produce
authentic Sebastian Michaelis dialogue.
"""
import logging
from typing import Dict, Any, Optional, List
import datetime

logger = logging.getLogger(__name__)

class DialogueManager:
    """
    Manages dialogue generation for Sebastian's personality.
    
    Integrates tone modulation, mannerisms, and emotion simulation
    to produce authentic Sebastian Michaelis dialogue patterns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize dialogue manager with configuration.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        
        # Initialize components
        from core.personality.tone_modulator import ToneModulator
        from core.personality.mannerisms import MannerismApplier
        from core.personality.emotion_simulator import EmotionSimulator
        
        self.tone_modulator = ToneModulator(self.config.get("tone", {}))
        self.mannerism_applier = MannerismApplier(self.config.get("mannerisms", {}))
        self.emotion_simulator = EmotionSimulator(self.config.get("emotions", {}))
        
        # Dialogue history for context awareness
        self.dialogue_history = []
        self.max_history_length = self.config.get("max_history_length", 10)
        
        logger.info("Dialogue manager initialized")
        
    def format_response(self, response_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Format a response according to Sebastian's personality.
        
        Args:
            response_data: Response data including content and metadata
            context: Dialogue context with user info, emotion, etc.
            
        Returns:
            Formatted response text
        """
        # Extract base content
        content = response_data.get("content", "")
        if not content:
            return ""
            
        # Extract relevant context
        relationship = context.get("relationship", "master")
        situation = context.get("situation", "neutral")
        emotion = response_data.get("emotion", context.get("emotion", "neutral"))
        emotion_intensity = context.get("emotion_intensity", 0.5)
        formality_level = context.get("formality_level", 8)
        time_of_day = context.get("time_of_day", self._get_time_of_day())
        
        # Process through personality layers
        
        # 1. First apply formal tone
        formal_text = self.tone_modulator.adjust_formality(content, formality_level)
        
        # 2. Then apply mannerisms
        text_with_mannerisms = self.mannerism_applier.apply(formal_text, relationship, situation)
        
        # 3. Finally apply emotional coloring (subtly)
        final_text = self.emotion_simulator.apply_emotion(text_with_mannerisms, emotion, emotion_intensity)
        
        # Special case: Add relationship-specific addressing
        final_text = self.tone_modulator.adjust_for_relationship(final_text, relationship)
        
        # Special case: For greetings, use time-specific formulations
        if response_data.get("intent") == "greeting":
            greeting = self.mannerism_applier.get_greeting(time_of_day, relationship)
            if greeting:
                final_text = greeting
                
        # Special case: For demon moments (rare)
        if context.get("demon_mode", False) or (response_data.get("intent") == "threat"):
            final_text = self.emotion_simulator.apply_demon_undertone(final_text, 0.7)
        
        # Update dialogue history
        self._update_history(final_text)
        
        return final_text
        
    def _update_history(self, text: str):
        """Update dialogue history with new text."""
        self.dialogue_history.append({
            "text": text,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Maintain maximum history length
        if len(self.dialogue_history) > self.max_history_length:
            self.dialogue_history = self.dialogue_history[-self.max_history_length:]
            
    def _get_time_of_day(self) -> str:
        """Get current time of day."""
        hour = datetime.datetime.now().hour
        
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "night"
            
    def get_response_for_intent(self, intent: str, context: Dict[str, Any]) -> str:
        """
        Get a response specifically for an intent.
        
        Args:
            intent: Intent to respond to
            context: Context information
            
        Returns:
            Response text
        """
        # Map intents to response types
        intent_map = {
            "greeting": lambda: self.mannerism_applier.get_greeting(
                context.get("time_of_day", self._get_time_of_day()),
                context.get("relationship", "master")
            ),
            "farewell": lambda: "I shall await your return, my lord.",
            "affirmation": lambda: self.tone_modulator.get_formal_response("affirmative", context),
            "negation": lambda: self.tone_modulator.get_formal_response("negative", context),
            "gratitude": lambda: "It is my duty to serve.",
            "apology": lambda: "I beg your forgiveness for any inconvenience caused.",
            "uncertain": lambda: self.tone_modulator.get_formal_response("uncertain", context),
            "threat": lambda: "I am simply one hell of a butler."
        }
        
        if intent in intent_map:
            response = intent_map[intent]()
            return self.format_response({"content": response, "intent": intent}, context)
            
        return ""