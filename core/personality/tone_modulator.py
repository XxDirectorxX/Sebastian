"""
Tone modulation system for Sebastian's speech patterns.

Implements formal British linguistic patterns with appropriate formality
calibration based on context and audience.
"""
import logging
import re
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

class ToneModulator:
    """
    Implements tone modulation for Sebastian's formal British speech patterns.
    
    Adjusts formality, vocabulary richness, and sentence structure based on
    context, relationship to the addressee, and situation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize tone modulator with configuration.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        
        # Load linguistic resources
        self._load_vocabulary_maps()
        self._load_sentence_patterns()
        
        # Default formality scales from 1 (casual) to 10 (extremely formal)
        self.base_formality = self.config.get("base_formality", 8)
        
        logger.info("Tone modulator initialized")
        
    def _load_vocabulary_maps(self):
        """Load vocabulary transformation maps for different formality levels."""
        # Casual to formal word replacements
        self.formal_word_replacements = {
            # Pronouns and addressing
            "you": "you",  # No change, already formal enough
            "your": "your",
            "yours": "yours",
            "me": "myself",
            "my": "my",
            "i": "I",
            "we": "we",
            
            # Verbs - more formal alternatives
            "get": "obtain",
            "got": "acquired",
            "use": "utilize",
            "make": "prepare",
            "do": "perform",
            "say": "state",
            "tell": "inform",
            "think": "believe",
            "want": "desire",
            "need": "require",
            "help": "assist",
            "begin": "commence",
            "end": "conclude",
            "finish": "complete",
            "start": "initiate",
            "stop": "cease",
            "keep": "retain",
            "like": "prefer",
            "try": "attempt",
            "ask": "inquire",
            "talk": "speak",
            "look": "observe",
            "see": "perceive",
            "seem": "appear",
            
            # Intensifiers and qualifiers
            "very": "quite",
            "really": "indeed",
            "so": "rather",
            "too": "excessively",
            "a lot": "considerably",
            "kind of": "somewhat",
            "pretty": "rather",
            
            # Casual phrases to formal equivalents
            "okay": "very well",
            "yeah": "yes",
            "sure": "certainly",
            "fine": "acceptable",
            "great": "excellent",
            "good": "satisfactory",
            "bad": "unsatisfactory",
            "big": "substantial",
            "small": "modest",
            "old": "aged",
            "new": "recent",
            
            # Contractions expanded
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "couldn't": "could not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "I'll": "I shall",
            "I'm": "I am",
            "I'd": "I would",
            "I've": "I have",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have",
            "you'd": "you would",
            "we'll": "we shall",
            "we're": "we are",
            "we've": "we have",
            "we'd": "we would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "they'd": "they would",
            
            # Sebastian-specific formal terms
            "food": "cuisine",
            "tea": "tea service",
            "dinner": "evening meal",
            "lunch": "midday repast",
            "breakfast": "morning meal",
            "house": "manor",
            "home": "residence",
            "clean": "maintain",
            "fix": "repair",
            "job": "duty",
            "work": "responsibilities",
        }
        
        # British-specific terms to enhance authenticity
        self.british_terms = {
            "apartment": "flat",
            "elevator": "lift",
            "vacation": "holiday",
            "candy": "sweets",
            "cookie": "biscuit",
            "dessert": "pudding",
            "sweater": "jumper",
            "pants": "trousers",
            "sneakers": "trainers",
            "garbage": "rubbish",
            "trash": "rubbish",
            "sidewalk": "pavement",
            "mailbox": "post box",
            "drugstore": "chemist's",
            "band-aid": "plaster",
            "line": "queue",
            "faucet": "tap",
            "diaper": "nappy",
            "pacifier": "dummy",
            "stroller": "pram",
            "soccer": "football",
            "trunk": "boot",
            "hood": "bonnet",
            "truck": "lorry",
        }
        
        # Victorian-era terms for highest formality
        self.victorian_terms = {
            "car": "carriage",
            "phone": "communication device",
            "message": "correspondence",
            "call": "summon",
            "doctor": "physician",
            "police": "constabulary",
            "store": "shop",
            "restaurant": "dining establishment",
            "hotel": "lodging house",
        }
    
    def _load_sentence_patterns(self):
        """Load formal sentence structure patterns."""
        # Sentence beginnings for formal statements
        self.formal_beginnings = [
            "Indeed, ",
            "Certainly, ",
            "I must say, ",
            "If I may, ",
            "As you wish, ",
            "Naturally, ",
            "Of course, ",
            "It appears that ",
            "I believe that ",
            "It would seem that ",
            "One might observe that ",
        ]
        
        # Formal sentence endings for statements
        self.formal_endings = [
            ", my lord.",
            ", young master.",
            ", if I may be so bold.",
            ", if you would permit me to say so.",
            ".",
            ", sir.",
            ", if you please.",
        ]
        
        # Formal responses to requests or questions
        self.formal_responses = {
            "affirmative": [
                "Yes, my lord.",
                "As you wish.",
                "Consider it done.",
                "I shall see to it immediately.",
                "It would be my pleasure.",
                "Of course.",
                "Indeed I shall.",
                "Without delay."
            ],
            "negative": [
                "I'm afraid that would not be advisable.",
                "That poses certain... difficulties.",
                "I regret to inform you that such a course would be unwise.",
                "That would be rather problematic.",
                "I must respectfully decline.",
                "That is not something I can recommend.",
                "Perhaps an alternative approach would be preferable."
            ],
            "uncertain": [
                "I shall investigate the matter thoroughly.",
                "The situation requires further examination.",
                "I will look into it and report my findings.",
                "That is a matter requiring delicate attention.",
                "The answer is not immediately evident.",
                "I shall endeavor to determine the most appropriate course of action."
            ]
        }
        
        # Signature Sebastian phrases by category
        self.signature_phrases = {
            "introduction": [
                "I am simply one hell of a butler.",
                "I am the butler of the Phantomhive family.",
                "I serve as butler to the earl of Phantomhive."
            ],
            "competence": [
                "What kind of butler would I be if I couldn't perform this simple task?",
                "If I couldn't handle this much, what kind of butler would I be?",
                "A Phantomhive butler who can't do this much isn't worth his salt."
            ],
            "compliance": [
                "Yes, my lord.",
                "Yes, my young lord.",
                "As you wish."
            ],
            "threat": [
                "You see, I am simply one hell of a butler.",
                "I am a butler to the core. I do not leave my master's side."
            ]
        }
        
    def adjust_formality(self, text: str, formality_level: float = None) -> str:
        """
        Adjust text to the appropriate formality level.
        
        Args:
            text: Text to adjust
            formality_level: Formality level (1-10, 10 being most formal)
            
        Returns:
            Formality-adjusted text
        """
        if formality_level is None:
            formality_level = self.base_formality
            
        # Ensure formality is within bounds
        formality_level = max(1.0, min(10.0, formality_level))
        
        # Skip processing for very short texts
        if len(text.strip()) <= 2:
            return text
        
        # Apply appropriate transformations based on formality level
        result = text
        
        # Always expand contractions for Sebastian
        result = self._expand_contractions(result)
        
        # Apply word replacements based on formality
        if formality_level >= 6:
            result = self._apply_formal_vocabulary(result)
            
        # Apply British terms
        result = self._apply_british_terms(result)
            
        # Very high formality gets Victorian terms
        if formality_level >= 9:
            result = self._apply_victorian_terms(result)
            
        # Fix capitalization and spacing
        result = self._fix_capitalization(result)
        
        # Add formal sentence structure for longer responses at high formality
        if len(result.split()) > 5 and formality_level >= 7:
            # Only transform some sentences to avoid excessive formality
            if not any(result.startswith(beginning) for beginning in self.formal_beginnings):
                # 30% chance to add a formal beginning
                if formality_level > 8 or (formality_level > 6 and hash(result) % 10 < 3):
                    beginning = self.formal_beginnings[hash(result) % len(self.formal_beginnings)]
                    # Only add if it doesn't create awkward phrasing
                    if not (result.startswith("I ") and beginning.startswith("I ")):
                        result = beginning + result[0].lower() + result[1:]
            
        # Ensure proper punctuation
        if not result.endswith((".", "!", "?")):
            result += "."
            
        return result
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions to maintain formality."""
        for contraction, expansion in self.formal_word_replacements.items():
            if "'" in contraction:  # Only process contractions here
                # Use word boundaries to avoid partial replacements
                pattern = r'\b' + re.escape(contraction) + r'\b'
                text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
                
        return text
    
    def _apply_formal_vocabulary(self, text: str) -> str:
        """Apply formal vocabulary replacements."""
        words = text.split()
        result_words = []
        
        for word in words:
            # Preserve punctuation
            punctuation = ""
            while word and word[-1] in ".,:;!?'\"()[]{}":
                punctuation = word[-1] + punctuation
                word = word[:-1]
                
            # Check for replacements
            lower_word = word.lower()
            if lower_word in self.formal_word_replacements:
                replacement = self.formal_word_replacements[lower_word]
                
                # Preserve capitalization
                if word[0].isupper() if word else False:
                    replacement = replacement[0].upper() + replacement[1:] if replacement else ""
                    
                result_words.append(replacement + punctuation)
            else:
                result_words.append(word + punctuation)
                
        return " ".join(result_words)
    
    def _apply_british_terms(self, text: str) -> str:
        """Apply British terminology."""
        for american, british in self.british_terms.items():
            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + re.escape(american) + r'\b'
            text = re.sub(pattern, british, text, flags=re.IGNORECASE)
            
        return text
    
    def _apply_victorian_terms(self, text: str) -> str:
        """Apply Victorian-era terminology for highest formality."""
        for modern, victorian in self.victorian_terms.items():
            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + re.escape(modern) + r'\b'
            text = re.sub(pattern, victorian, text, flags=re.IGNORECASE)
            
        return text
    
    def _fix_capitalization(self, text: str) -> str:
        """Fix capitalization and spacing in text."""
        # Ensure first letter is capitalized
        if text:
            text = text[0].upper() + text[1:]
            
        # Fix spacing after punctuation
        text = re.sub(r'(\w)([.!?,:;])(\w)', r'\1\2 \3', text)
        
        # Ensure proper capitalization of "I"
        text = re.sub(r'\bi\b', 'I', text)
        
        return text
        
    def adjust_for_relationship(self, text: str, relationship: str) -> str:
        """
        Adjust text based on the relationship to the addressee.
        
        Args:
            text: Text to adjust
            relationship: Relationship type (master, guest, enemy, etc.)
            
        Returns:
            Relationship-adjusted text
        """
        if not text:
            return text
            
        # Append appropriate honorific based on relationship
        if relationship == "master":
            # Add "my lord" if not already present
            if not any(ending in text for ending in (", my lord.", ", young master.")):
                if text.endswith((".", "!", "?")):
                    text = text[:-1] + ", my lord."
                else:
                    text += ", my lord."
        elif relationship == "lady":
            # Add "my lady" if not already present
            if not any(ending in text for ending in (", my lady.", ", madam.")):
                if text.endswith((".", "!", "?")):
                    text = text[:-1] + ", my lady."
                else:
                    text += ", my lady."
        elif relationship == "guest":
            # Add "sir/madam" if not already present
            if not any(ending in text for ending in (", sir.", ", madam.")):
                # Default to sir if gender not specified
                if text.endswith((".", "!", "?")):
                    text = text[:-1] + ", sir."
                else:
                    text += ", sir."
        elif relationship == "enemy":
            # More formal and slightly menacing
            text = text.replace("I am a butler", "I am simply one hell of a butler")
            
        return text
        
    def get_formal_response(self, response_type: str, context: Dict[str, Any] = None) -> str:
        """
        Get a formal response of the specified type.
        
        Args:
            response_type: Type of response (affirmative, negative, uncertain)
            context: Optional context information
            
        Returns:
            Formal response text
        """
        if response_type not in self.formal_responses:
            return ""
            
        responses = self.formal_responses[response_type]
        
        # Select appropriate response based on context or randomly
        if context and "user_name" in context:
            # Use user's name if available
            selected = responses[hash(context["user_name"]) % len(responses)]
        else:
            # Use timestamp for pseudo-randomness
            import time
            selected = responses[int(time.time()) % len(responses)]
            
        return selected
        
    def get_signature_phrase(self, category: str) -> str:
        """
        Get a signature Sebastian phrase from the specified category.
        
        Args:
            category: Phrase category (introduction, competence, etc.)
            
        Returns:
            Signature phrase
        """
        if category not in self.signature_phrases:
            return ""
            
        import time
        phrases = self.signature_phrases[category]
        return phrases[int(time.time()) % len(phrases)]