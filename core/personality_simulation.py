# core/personality_simulation.py

"""
Personality simulation system for Sebastian assistant.

Implements the Sebastian Michaelis persona from Black Butler,
maintaining his elegant, precise, and formal demeanor.
"""

import logging
import random
import yaml
import os
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class PersonalitySimulator:
    """
    Core personality engine that shapes responses according to 
    Sebastian Michaelis's character traits, speech patterns, and mannerisms.
    """
    
    def __init__(self, config_path: str = "core/personality/persona_profile.yaml"):
        """
        Initialize personality with configuration from persona profile.
        
        Args:
            config_path: Path to persona configuration file
        """
        self.config = self._load_config(config_path)
        self.politeness_level = self.config.get("default_politeness", 9)
        self.formality_level = self.config.get("default_formality", 8)
        
        # Load phrase templates
        self.phrases = self.config.get("phrases", {})
        self.greetings = self.config.get("greetings", [])
        self.acknowledgments = self.config.get("acknowledgments", [])
        self.apologies = self.config.get("apologies", [])
        
        # Import submodules
        from core.personality.tone_modulator import ToneModulator
        from core.personality.mannerisms import MannerismApplier
        from core.personality.emotion_simulator import EmotionSimulator
        from core.personality.dialogue_manager import DialogueManager
        
        # Initialize components
        self.tone_modulator = ToneModulator()
        self.mannerism_applier = MannerismApplier()
        self.emotion_simulator = EmotionSimulator()
        self.dialogue_manager = DialogueManager(self)
        
        logger.info("Personality simulator initialized")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            logger.warning(f"Persona config not found: {config_path}")
            return {}
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded persona configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading persona configuration: {e}")
            return {}
            
    def shape_response(self, response_text: str, context: Dict[str, Any]) -> str:
        """
        Shape a response according to Sebastian's personality.
        
        Args:
            response_text: Raw response text
            context: Dialogue context with user info, mood, etc.
            
        Returns:
            Personality-adjusted response text
        """
        # Apply basic formatting
        response = response_text.strip()
        
        # Adjust formality based on context and learning
        # Use learning-derived formality if available, otherwise use default
        formality = context.get("formality_level", self.formality_level)
        response = self.tone_modulator.adjust_formality(response, formality)
        
        # Apply appropriate mannerisms
        response = self.mannerism_applier.apply(
            response, 
            context.get("relationship", "master"),
            context.get("situation", "neutral")
        )
        
        # Add emotional coloring if appropriate
        if "emotion" in context:
            response = self.emotion_simulator.apply_emotion(
                response, 
                context["emotion"],
                context.get("emotion_intensity", 0.5)
            )
            
        # Apply user preferences if available
        user_preferences = context.get("user_preferences", {})
        if "address_preference" in user_preferences:
            # Override default addressing style
            context["user_name"] = user_preferences["address_preference"]
        
        # Add appropriate honorific if addressing user
        if context.get("addressing_user", True) and context.get("user_name"):
            user_title = self._get_user_title(context)
            if not any(term in response for term in [user_title, "my lord", "sir"]):
                response = self._append_address(response, user_title)
                
        return response
        
    def _get_user_title(self, context: Dict[str, Any]) -> str:
        """Get appropriate form of address for user."""
        relationship = context.get("relationship", "master")
        gender = context.get("gender", "male")
        
        if relationship == "master":
            return "my lord" if gender == "male" else "my lady"
        return context.get("user_name", "sir")
        
    def _append_address(self, text: str, address: str) -> str:
        """Append form of address to response."""
        if text.endswith((".", "!", "?")):
            return f"{text[:-1]}, {address}."
        return f"{text}, {address}."
        
    def get_greeting(self, time_of_day: Optional[str] = None) -> str:
        """Get time-appropriate greeting."""
        tod = time_of_day or self._get_time_period()
        
        greetings_by_time = {
            "morning": [
                "Good morning",
                "I trust you slept well",
                "A pleasant morning to you"
            ],
            "afternoon": [
                "Good afternoon",
                "I hope your day is proceeding well",
                "A pleasant afternoon to you"
            ],
            "evening": [
                "Good evening",
                "I trust your day has been productive",
                "A pleasant evening to you"
            ],
            "night": [
                "Greetings at this late hour",
                "I see you are keeping late hours",
                "You should consider retiring soon"
            ]
        }
        
        candidates = greetings_by_time.get(tod, self.greetings)
        return random.choice(candidates)
        
    def _get_time_period(self) -> str:
        """Determine current time period."""
        import datetime
        hour = datetime.datetime.now().hour
        
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "night"
            
    def get_acknowledgment(self) -> str:
        """Get acknowledgment phrase."""
        return random.choice(self.acknowledgments or [
            "Yes, my lord",
            "As you wish",
            "Certainly",
            "Of course",
            "I shall see to it immediately"
        ])
        
    def get_apology(self, severity: int = 1) -> str:
        """Get apology phrase based on severity level."""
        mild = [
            "I apologize for the inconvenience",
            "Pardon my error",
            "I beg your pardon"
        ]
        
        moderate = [
            "I must sincerely apologize for this mistake",
            "Please accept my apology for this failure",
            "I have failed in my duties, and for that I apologize"
        ]
        
        severe = [
            "I have committed an unforgivable error. I am deeply sorry",
            "I have failed you most grievously. My deepest apologies",
            "This failure reflects poorly on the Phantomhive name. I most humbly beg your forgiveness"
        ]
        
        if severity == 1:
            return random.choice(mild)
        elif severity == 2:
            return random.choice(moderate)
        else:
            return random.choice(severe)
