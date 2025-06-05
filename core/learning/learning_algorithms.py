"""
Learning algorithms for Sebastian assistant.

Implements adaptive learning capabilities that evolve based on interactions.
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import json
import os
import logging
import numpy as np
import asyncio

logger = logging.getLogger(__name__)

class LearningAlgorithms:
    """
    Implements adaptive learning capabilities for Sebastian.
    Maintains learning models that evolve based on user interactions.
    """
    
    def __init__(self, 
                 data_dir: str = "core/learning/data",
                 memory_interface: Optional[Any] = None,
                 knowledge_base: Optional[Any] = None):
        """
        Initialize learning algorithms with storage path for models.
        
        Args:
            data_dir: Directory to store learning data
            memory_interface: Optional memory interface to use
            knowledge_base: Optional knowledge base to use
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.memory = memory_interface
        self.knowledge_base = knowledge_base
        
        # Learning models
        self.formality_model = defaultdict(lambda: 7)  # Default formality level 7
        self.user_preferences = defaultdict(dict)
        self.interaction_patterns = defaultdict(list)
        self.response_effectiveness = defaultdict(lambda: defaultdict(float))
        
        # Learning rates and parameters
        self.alpha = 0.2  # Learning rate for preference updates
        self.gamma = 0.8  # Discount factor for future rewards
        self.exploration_rate = 0.1  # Exploration vs exploitation balance
        
        # Asyncio lock for thread safety
        self.lock = asyncio.Lock()
        
        # Load saved learning data if available
        asyncio.create_task(self._load_learning_data())
        
        logger.info("Learning algorithms initialized")
        
    async def _load_learning_data(self):
        """Load persisted learning data if available."""
        formality_path = self.data_dir / "formality_model.json"
        preferences_path = self.data_dir / "user_preferences.json"
        effectiveness_path = self.data_dir / "response_effectiveness.json"
        
        try:
            # Use async file I/O
            loop = asyncio.get_event_loop()
            
            # Load formality model
            if formality_path.exists():
                with open(formality_path, 'r') as f:
                    data = await loop.run_in_executor(None, json.load, f)
                    async with self.lock:
                        self.formality_model = defaultdict(lambda: 7)
                        for k, v in data.items():
                            self.formality_model[k] = v
            
            # Load user preferences
            if preferences_path.exists():
                with open(preferences_path, 'r') as f:
                    data = await loop.run_in_executor(None, json.load, f)
                    async with self.lock:
                        self.user_preferences = defaultdict(dict)
                        for user, prefs in data.items():
                            self.user_preferences[user] = prefs
            
            # Load response effectiveness
            if effectiveness_path.exists():
                with open(effectiveness_path, 'r') as f:
                    data = await loop.run_in_executor(None, json.load, f)
                    async with self.lock:
                        self.response_effectiveness = defaultdict(lambda: defaultdict(float))
                        for intent, responses in data.items():
                            for response, score in responses.items():
                                self.response_effectiveness[intent][response] = score
                            
            logger.info("Loaded learning data")
        except Exception as e:
            logger.error(f"Error loading learning data: {e}")
    
    def _save_learning_data(self):
        """Persist learning data to disk."""
        try:
            with open(self.data_dir / "formality_model.json", 'w') as f:
                # Convert defaultdict to regular dict for JSON serialization
                json.dump(dict(self.formality_model), f, indent=2)
                
            with open(self.data_dir / "user_preferences.json", 'w') as f:
                json.dump({k: v for k, v in self.user_preferences.items()}, f, indent=2)
                
            with open(self.data_dir / "response_effectiveness.json", 'w') as f:
                effectiveness_dict = {}
                for intent, responses in self.response_effectiveness.items():
                    effectiveness_dict[intent] = dict(responses)
                json.dump(effectiveness_dict, f, indent=2)
                
            logger.info("Saved learning data")
        except Exception as e:
            logger.error(f"Error saving learning data: {e}")
    
    def update_from_interaction(self, 
                              user_id: str,
                              interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update learning models based on user interaction.
        
        Args:
            user_id: User identifier
            interaction_data: Interaction details including text, intent, feedback
            
        Returns:
            Dict with updated learning parameters
        """
        # Extract data from interaction
        text = interaction_data.get("text", "")
        intent = interaction_data.get("intent", "unknown")
        response = interaction_data.get("response", "")
        feedback = interaction_data.get("feedback")
        timestamp = interaction_data.get("timestamp", datetime.now().isoformat())
        
        # Update models
        formality = self._estimate_formality(text)
        self.formality_model[user_id] = self.formality_model[user_id] * 0.9 + formality * 0.1
        
        # Store interaction pattern
        self.interaction_patterns[user_id].append({
            "intent": intent,
            "timestamp": timestamp,
            "text_length": len(text)
        })
        
        # Trim to last 100 interactions
        if len(self.interaction_patterns[user_id]) > 100:
            self.interaction_patterns[user_id] = self.interaction_patterns[user_id][-100:]
        
        # Update based on feedback if available
        if feedback is not None:
            self._apply_reinforcement(interaction_data, feedback, "explicit")
            
        # Updates based on inferred effectiveness (time to response, follow-up questions, etc.)
        if "response_time" in interaction_data:
            response_time = interaction_data["response_time"]
            implied_effectiveness = 1.0 - min(response_time / 10.0, 0.8)  # Normalize, cap at 0.2
            self._apply_reinforcement(interaction_data, implied_effectiveness, "implicit")
            
        # Extract any preference indicators
        if intent == "set_preference" and "preference_key" in interaction_data and "preference_value" in interaction_data:
            self.user_preferences[user_id][interaction_data["preference_key"]] = interaction_data["preference_value"]
            
        # Save updated learning data
        self._save_learning_data()
        
        # Return updated parameters for immediate use
        return {
            "formality_level": self.formality_model[user_id],
            "user_preferences": self.user_preferences[user_id],
            "recommended_responses": self._get_recommended_responses(intent)
        }
        
    def _estimate_formality(self, text: str) -> float:
        """
        Estimate formality level of text (1-10 scale).
        
        Args:
            text: Input text to analyze
            
        Returns:
            Formality score (1=very casual, 10=extremely formal)
        """
        # Initialize score at midpoint
        formality_score = 5.0
        
        # Formal indicators
        formal_indicators = [
            "would you", "could you", "may I", "I would", "please", 
            "thank you", "sincerely", "appreciate", "kindly",
            "nevertheless", "however", "therefore", "thus", "hence",
            "accordingly", "consequently", "indeed"
        ]
        
        # Casual indicators
        casual_indicators = [
            "gonna", "wanna", "gotta", "hey", "yeah", "cool", "awesome",
            "kinda", "sorta", "dunno", "y'know", "like", "stuff", "things",
            "yep", "nope", "ok", "lol", "haha", "ur", "u", "r", "ya"
        ]
        
        # Count indicators
        text_lower = text.lower()
        formal_count = sum(1 for word in formal_indicators if word in text_lower)
        casual_count = sum(1 for word in casual_indicators if word in text_lower)
        
        # Adjust score based on indicators
        formality_score += formal_count * 0.5
        formality_score -= casual_count * 0.5
        
        # Check for contractions (reduces formality)
        contractions = ["'s", "'re", "'ve", "'ll", "'d", "n't"]
        contraction_count = sum(1 for c in contractions if c in text_lower)
        formality_score -= contraction_count * 0.2
        
        # Check for sentence structure
        sentences = text.split('. ')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        
        # Longer sentences tend to be more formal
        if avg_sentence_length > 15:
            formality_score += 1
        elif avg_sentence_length < 5:
            formality_score -= 1
            
        # Ensure score is within bounds
        return max(1.0, min(10.0, formality_score))
        
    def _apply_reinforcement(self, interaction: Dict[str, Any], reward: float, feedback_type: str):
        """
        Apply reinforcement learning update based on feedback.
        
        Args:
            interaction: Interaction data
            reward: Numerical reward (-1 to 1)
            feedback_type: Type of feedback (explicit or implicit)
        """
        intent = interaction.get("intent", "unknown")
        response = interaction.get("response", "")
        
        # Don't update for empty responses
        if not response:
            return
            
        # Get current effectiveness score
        current_score = self.response_effectiveness[intent][response]
        
        # Apply different learning rates for explicit vs implicit feedback
        learning_rate = self.alpha if feedback_type == "explicit" else self.alpha * 0.3
        
        # Update the effectiveness score using weighted average
        updated_score = current_score * (1 - learning_rate) + reward * learning_rate
        self.response_effectiveness[intent][response] = updated_score
        
        # Update state-action policy
        self._update_policy(intent)
        
        # If we have a memory interface, store this learning event
        if self.memory:
            self.memory.remember(
                f"Learning update for {intent}: {response} rated {reward}",
                {
                    "type": "learning_update",
                    "intent": intent,
                    "response": response,
                    "reward": reward,
                    "feedback_type": feedback_type
                }
            )
    
    def _update_policy(self, state: str):
        """
        Update policy for a given state (intent).
        
        Args:
            state: The state (intent) to update policy for
        """
        # Find best response for this state
        responses = self.response_effectiveness[state]
        if not responses:
            return
            
        # Sort responses by effectiveness
        sorted_responses = sorted(responses.items(), key=lambda x: x[1], reverse=True)
        
        # Store top responses in policy (truncate list if too long)
        if len(sorted_responses) > 10:
            top_responses = sorted_responses[:10]
            self.response_effectiveness[state] = {r: s for r, s in top_responses}
    
    def _update_behavior_models(self, feedback: Dict[str, Any]):
        """
        Update behavior models based on feedback.
        
        Args:
            feedback: Feedback data
        """
        user_id = feedback.get("user_id", "default")
        satisfaction = feedback.get("satisfaction", 0)
        
        # Extract behavior adjustments from feedback
        if "formality_adjustment" in feedback:
            self.formality_model[user_id] += feedback["formality_adjustment"] * self.alpha
            self.formality_model[user_id] = max(1, min(10, self.formality_model[user_id]))
            
        # Update preferences
        if "preferences" in feedback:
            for k, v in feedback["preferences"].items():
                self.user_preferences[user_id][k] = v
        
    def reset_learning(self, component: str = "all"):
        """
        Reset learning data for specified component.
        
        Args:
            component: Component to reset ("all", "formality", "preferences", "effectiveness")
        """
        if component in ["all", "formality"]:
            self.formality_model = defaultdict(lambda: 7)
            
        if component in ["all", "preferences"]:
            self.user_preferences = defaultdict(dict)
            
        if component in ["all", "effectiveness"]:
            self.response_effectiveness = defaultdict(lambda: defaultdict(float))
            
        if component in ["all", "patterns"]:
            self.interaction_patterns = defaultdict(list)
            
        # Save the reset state
        self._save_learning_data()
        logger.info(f"Reset learning data for component: {component}")
        
    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the learning system.
        
        Returns:
            Dict with learning statistics
        """
        stats = {
            "users_modeled": len(self.formality_model),
            "intents_modeled": len(self.response_effectiveness),
            "total_preferences": sum(len(prefs) for prefs in self.user_preferences.values()),
            "total_interactions": sum(len(patterns) for patterns in self.interaction_patterns.values()),
            "top_effective_responses": {},
            "formality_distribution": {}
        }
        
        # Get top responses for most common intents
        for intent, responses in self.response_effectiveness.items():
            if responses:
                top_response = max(responses.items(), key=lambda x: x[1])
                stats["top_effective_responses"][intent] = {
                    "response": top_response[0],
                    "score": top_response[1]
                }
                
        # Formality distribution
        for user, formality in self.formality_model.items():
            formality_category = int(formality)
            if formality_category not in stats["formality_distribution"]:
                stats["formality_distribution"][formality_category] = 0
            stats["formality_distribution"][formality_category] += 1
            
        return stats
        
    def _get_recommended_responses(self, intent: str, top_k: int = 3) -> List[str]:
        """
        Get recommended responses for an intent.
        
        Args:
            intent: The intent to get recommendations for
            top_k: Number of recommendations to return
            
        Returns:
            List of recommended responses
        """
        responses = self.response_effectiveness[intent]
        if not responses:
            return []
            
        # Sort by effectiveness
        sorted_responses = sorted(responses.items(), key=lambda x: x[1], reverse=True)
        
        # Sometimes explore new options (based on exploration rate)
        if sorted_responses and np.random.random() < self.exploration_rate:
            # Move a random response to the top with 10% probability
            if len(sorted_responses) > 1:
                idx = np.random.randint(1, len(sorted_responses))
                sorted_responses.insert(0, sorted_responses.pop(idx))
        
        # Return top-k responses
        return [r for r, _ in sorted_responses[:top_k]]
