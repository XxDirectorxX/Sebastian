"""
Emotion simulation system for Sebastian's restrained emotional expressions.

Implements subtle variations in language based on emotional state,
while maintaining the calm, composed demeanor characteristic of Sebastian.
"""
import logging
import random
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class EmotionSimulator:
    """
    Simulates Sebastian's restrained emotional expressions.
    
    Applies subtle modifications to language based on emotional state,
    while ensuring the butler's characteristic restraint and composure.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize emotion simulator with configuration.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        
        # Load emotion resources
        self._load_emotion_resources()
        
        # Default emotional restraint (0-1, higher means more restraint)
        self.emotional_restraint = self.config.get("emotional_restraint", 0.8)
        
        logger.info("Emotion simulator initialized")
        
    def _load_emotion_resources(self):
        """Load emotion expression resources."""
        # Emotion intensity vocabulary modifiers
        self.emotion_vocabulary = {
            # Positive emotions
            "pleased": {
                "adverbs": ["adequately", "suitably", "quite", "rather"],
                "adjectives": ["satisfactory", "acceptable", "proper", "appropriate"],
                "verbs": ["appreciate", "approve", "commend", "acknowledge"]
            },
            "satisfied": {
                "adverbs": ["quite", "rather", "indeed", "certainly"],
                "adjectives": ["pleasing", "gratifying", "agreeable", "favorable"],
                "verbs": ["approve", "commend", "endorse", "praise"]
            },
            "amused": {
                "adverbs": ["mildly", "somewhat", "slightly", "faintly"],
                "adjectives": ["diverting", "entertaining", "curious", "peculiar"],
                "verbs": ["observe", "note", "perceive", "find"]
            },
            
            # Negative emotions (always understated for Sebastian)
            "displeased": {
                "adverbs": ["rather", "somewhat", "unfortunately", "regrettably"],
                "adjectives": ["unsatisfactory", "inappropriate", "unbecoming", "unsuitable"],
                "verbs": ["disapprove", "question", "doubt", "discourage"]
            },
            "concerned": {
                "adverbs": ["somewhat", "slightly", "reasonably", "justifiably"],
                "adjectives": ["concerning", "troubling", "worrying", "questionable"],
                "verbs": ["observe", "note", "consider", "find"]
            },
            "irritated": {
                "adverbs": ["quite", "rather", "notably", "decidedly"],
                "adjectives": ["tiresome", "vexing", "trying", "troublesome"],
                "verbs": ["find", "consider", "deem", "regard"]
            },
            
            # Rare stronger emotions (used very sparingly)
            "angry": {
                "adverbs": ["decidedly", "unquestionably", "unmistakably", "absolutely"],
                "adjectives": ["unacceptable", "intolerable", "inexcusable", "impermissible"],
                "verbs": ["insist", "demand", "require", "command"]
            },
            "protective": {
                "adverbs": ["immediately", "urgently", "directly", "explicitly"],
                "adjectives": ["necessary", "imperative", "essential", "critical"],
                "verbs": ["protect", "defend", "safeguard", "secure"]
            }
        }
        
        # Emotion-specific phrases
        self.emotion_phrases = {
            "pleased": [
                "How very satisfactory.",
                "This is most agreeable.",
                "I find this outcome quite acceptable."
            ],
            "satisfied": [
                "Everything is proceeding as it should.",
                "The situation has been resolved appropriately.",
                "All matters have been attended to properly."
            ],
            "amused": [
                "How curious.",
                "Most entertaining.",
                "What an interesting development."
            ],
            "displeased": [
                "This is rather disappointing.",
                "I find this somewhat unsatisfactory.",
                "The outcome is not quite what one would hope for."
            ],
            "concerned": [
                "This requires immediate attention.",
                "The situation may be problematic.",
                "We must address this matter with care."
            ],
            "irritated": [
                "This is becoming rather tiresome.",
                "I find this development quite vexing.",
                "The circumstances are increasingly trying."
            ],
            "angry": [
                "This is entirely unacceptable.",
                "I must insist that this ceases immediately.",
                "Such behavior cannot be tolerated."
            ],
            "protective": [
                "Please remain behind me, my lord.",
                "Your safety is my utmost priority.",
                "I shall not allow any harm to come to you."
            ]
        }
        
        # Emotional tone markers (subtle linguistic patterns for each emotion)
        self.emotional_markers = {
            "pleased": {
                "patterns": [
                    "I am {adverb} {adjective} with this",
                    "This is {adverb} {adjective}",
                    "I {verb} of this outcome"
                ],
                "punctuation": ".",
            },
            "satisfied": {
                "patterns": [
                    "This is {adverb} {adjective}",
                    "I {verb} this result",
                    "The outcome is {adverb} {adjective}"
                ],
                "punctuation": ".",
            },
            "amused": {
                "patterns": [
                    "How {adjective}",
                    "I {verb} this {adverb} {adjective}",
                    "What a {adjective} situation"
                ],
                "punctuation": ".",
            },
            "displeased": {
                "patterns": [
                    "This is {adverb} {adjective}",
                    "I {verb} of such matters",
                    "I find this {adverb} {adjective}"
                ],
                "punctuation": ".",
            },
            "concerned": {
                "patterns": [
                    "This is {adverb} {adjective}",
                    "I {verb} this development {adjective}",
                    "The situation is {adjective}"
                ],
                "punctuation": ".",
            },
            "irritated": {
                "patterns": [
                    "This is {adverb} {adjective}",
                    "I {verb} this {adjective}",
                    "Such behavior is {adverb} {adjective}"
                ],
                "punctuation": ".",
            },
            "angry": {
                "patterns": [
                    "This is {adverb} {adjective}",
                    "I must {verb} that this ceases",
                    "Such conduct is {adverb} {adjective}"
                ],
                "punctuation": ".",
            },
            "protective": {
                "patterns": [
                    "It is {adverb} {adjective} that you remain safe",
                    "I shall {verb} you",
                    "Your safety is {adverb} {adjective}"
                ],
                "punctuation": ".",
            }
        }
    
    def apply_emotion(self, text: str, emotion: str, intensity: float = 0.5) -> str:
        """
        Apply emotional coloring to text.
        
        Args:
            text: Text to modify
            emotion: Emotion to apply
            intensity: Emotion intensity (0-1)
            
        Returns:
            Emotionally colored text
        """
        # Return original text for very short inputs
        if len(text.strip()) <= 3:
            return text
            
        # Apply emotional restraint to intensity
        restrained_intensity = intensity * (1 - self.emotional_restraint)
        
        # Skip if emotion not recognized or intensity too low after restraint
        if emotion not in self.emotion_vocabulary or restrained_intensity < 0.1:
            return text
            
        # Determine if we should apply emotion based on restraint and intensity
        # Sebastian rarely shows strong emotion, so high restraint or low intensity may result in no change
        if random.random() > restrained_intensity:
            return text
            
        result = text
        
        # For very high intensity (and passed restraint check), consider adding emotional phrase
        if intensity > 0.7 and random.random() < restrained_intensity:
            if emotion in self.emotion_phrases:
                emotional_phrase = random.choice(self.emotion_phrases[emotion])
                
                # Add phrase at beginning or end based on emotion
                if emotion in ["concerned", "angry", "protective"]:
                    # More urgent emotions tend to be expressed first
                    result = emotional_phrase + " " + result
                else:
                    # Reflective emotions tend to come after the content
                    result = result.rstrip()
                    if result.endswith((".", "!", "?")):
                        result = result + " " + emotional_phrase
                    else:
                        result = result + ". " + emotional_phrase
        
        # For lower intensity, apply subtle linguistic markers
        else:
            # Modify tone through word choice
            result = self._apply_emotional_tone(result, emotion, restrained_intensity)
            
            # For certain emotions, modify punctuation
            result = self._apply_emotional_punctuation(result, emotion, restrained_intensity)
        
        return result
    
    def _apply_emotional_tone(self, text: str, emotion: str, intensity: float) -> str:
        """Apply emotional tone through word choice modifications."""
        # Get vocabulary for this emotion
        vocab = self.emotion_vocabulary.get(emotion, {})
        
        # Get random selections from vocabulary
        adverb = random.choice(vocab.get("adverbs", ["quite"]))
        adjective = random.choice(vocab.get("adjectives", ["interesting"]))
        verb = random.choice(vocab.get("verbs", ["find"]))
        
        # For subtle emotional coloring, we'll only modify if the text doesn't already
        # contain strong emotional markers
        words = text.split()
        
        # For medium-length texts, consider adding emotional marker based on intensity
        if 5 <= len(words) <= 20 and random.random() < intensity:
            # Select a pattern and format it with our vocabulary
            if emotion in self.emotional_markers:
                patterns = self.emotional_markers[emotion]["patterns"]
                pattern = random.choice(patterns)
                
                emotional_marker = pattern.format(
                    adverb=adverb,
                    adjective=adjective,
                    verb=verb
                )
                
                # Add marker at beginning or end based on emotion type
                if emotion in ["concerned", "angry", "protective"]:
                    # Urgent emotions expressed first
                    return emotional_marker + " " + text
                else:
                    # Reflective emotions come after
                    if text.endswith((".", "!", "?")):
                        return text + " " + emotional_marker
                    else:
                        return text + ". " + emotional_marker
        
        return text
    
    def _apply_emotional_punctuation(self, text: str, emotion: str, intensity: float) -> str:
        """Apply emotional punctuation modifications."""
        # Different emotions may affect punctuation differently
        if emotion in ["angry", "protective"] and intensity > 0.6:
            # More forceful punctuation
            if text.endswith("."):
                text = text[:-1] + ".":
            elif not text.endswith(("!", "?")):
                text = text + "."
        elif emotion in ["amused"] and intensity > 0.5:
            # More expressive punctuation
            if text.endswith("."):
                text = text[:-1] + ".":
            elif not text.endswith(("!", "?")):
                text = text + "."
                
        return text
        
    def apply_demon_undertone(self, text: str, intensity: float = 0.3) -> str:
        """
        Apply subtle demonic undertones to text.
        
        Args:
            text: Text to modify
            intensity: Undertone intensity
            
        Returns:
            Text with demonic undertones
        """
        # Skip for short texts or low intensity
        if len(text.strip()) <= 10 or intensity <= 0.1:
            return text
            
        # Demonic phrases that could be inserted
        demonic_phrases = [
            "I am, after all, one hell of a butler.",
            "As a Phantomhive butler, I can manage at least this much.",
            "My master's orders are absolute.",
            "A butler who cannot accomplish this much isn't worth his salt.",
            "I never lie.",
            "Yes, my lord.",
            "I shall be at your side until the very end."
        ]
        
        # Apply with probability based on intensity
        if random.random() < intensity:
            phrase = random.choice(demonic_phrases)
            
            # Add phrase at end of text
            result = text.rstrip()
            if result.endswith((".", "!", "?")):
                result = result + " " + phrase
            else:
                result = result + ". " + phrase
                
            return result
            
        return text