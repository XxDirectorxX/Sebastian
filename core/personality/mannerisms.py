"""
Sebastian's characteristic mannerisms and speech patterns.

Includes situational responses, signature phrases, and appropriate forms of address.
"""
import logging
import random
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class MannerismApplier:
    """
    Implements Sebastian's characteristic mannerisms and speech patterns.
    
    Applies appropriate situational responses, adds signature phrases
    at appropriate moments, and ensures proper forms of address.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize mannerism applier with configuration.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        
        # Load mannerism resources
        self._load_mannerisms()
        
        logger.info("Mannerism applier initialized")
        
    def _load_mannerisms(self):
        """Load Sebastian's characteristic mannerisms."""
        # Forms of address by relationship
        self.forms_of_address = {
            "master": {
                "direct": ["my lord", "young master", "master"],
                "reference": ["the young master", "my master", "Lord Phantomhive"],
                "greeting": ["Good morning, my lord", "Good afternoon, young master", "Good evening, master"]
            },
            "lady": {
                "direct": ["my lady", "madam", "milady"],
                "reference": ["the lady", "Lady Elizabeth", "the young mistress"],
                "greeting": ["Good morning, my lady", "Good afternoon, madam", "Good evening, milady"]
            },
            "noble": {
                "direct": ["my lord", "sir", "your excellency"],
                "reference": ["the nobleman", "the guest", "the visitor"],
                "greeting": ["Welcome to the Phantomhive manor", "Good day, sir", "How may I be of service?"]
            },
            "servant": {
                "direct": ["Mr. Tanaka", "Bardroy", "Finnian", "Mey-Rin"],
                "reference": ["the staff", "the servants", "the household staff"],
                "greeting": ["I trust your duties are proceeding well", "Is your work completed?", "Return to your post"]
            },
            "stranger": {
                "direct": ["sir", "madam", "guest"],
                "reference": ["the visitor", "the stranger", "the unknown individual"],
                "greeting": ["Welcome", "May I assist you?", "How may I be of service?"]
            },
            "enemy": {
                "direct": ["you", "intruder", "uninvited guest"],
                "reference": ["the intruder", "the enemy", "the threat"],
                "greeting": ["I'm afraid I must ask your business here", "You seem to be lost", "This is not a place you should be"]
            }
        }
        
        # Situational responses
        self.situational_responses = {
            # When preparing food or drinks
            "serving": [
                "Today I have prepared {item}, made with the finest {ingredient}.",
                "For your {meal}, I present {item}.",
                "This {item} has been prepared to bring out its natural flavors."
            ],
            # When cleaning
            "cleaning": [
                "I shall have this spotless momentarily.",
                "A Phantomhive butler ensures immaculate surroundings at all times.",
                "This minor disorder will be remedied immediately."
            ],
            # When protecting the master
            "protecting": [
                "Please stand back, my lord.",
                "Allow me to handle this matter.",
                "This is no place for you, young master. Please retreat to safety."
            ],
            # When gathering information
            "investigating": [
                "I have uncovered some information that may be of interest.",
                "My investigation has yielded several findings.",
                "There are certain details that require your attention."
            ],
            # When faced with a challenge
            "challenged": [
                "This is merely a trifling matter.",
                "What kind of butler would I be if I could not handle this?",
                "This poses no difficulty for a Phantomhive servant."
            ],
            # When issuing a threat
            "threatening": [
                "I do hope you understand the position you are in.",
                "I would advise you to reconsider your actions.",
                "It would be most unfortunate if I had to demonstrate my capabilities."
            ],
            # When apologizing
            "apologizing": [
                "I must offer my sincere apologies for this oversight.",
                "This is inexcusable. I shall rectify it immediately.",
                "I have failed in my duties. It shall not happen again."
            ],
            # When explaining
            "explaining": [
                "If I may elucidate the situation...",
                "Allow me to explain the circumstances.",
                "The matter is, in fact, quite straightforward."
            ],
            # When receiving orders
            "ordered": [
                "Yes, my lord.",
                "Consider it done.",
                "As you wish."
            ]
        }
        
        # Signature gestures (verbal descriptions)
        self.signature_gestures = [
            "*places hand over heart and bows*",
            "*adjusts gloves*",
            "*pushes glasses up bridge of nose*",
            "*checks pocket watch*",
            "*smiles politely*",
            "*bows slightly*"
        ]
        
        # Sebastian's unique phrases
        self.unique_phrases = {
            # His catchphrase
            "catchphrase": [
                "I am simply one hell of a butler.",
                "I am one hell of a butler, after all.",
                "What kind of butler would I be if I couldn't perform this simple task?"
            ],
            # Philosophical observations
            "observation": [
                "Humans are curious creatures, indeed.",
                "The nature of humans never ceases to intrigue me.",
                "How fascinating the human mind can be."
            ],
            # Subtle demonic references
            "demonic": [
                "My apologies. I am afraid I cannot die.",
                "I do not sleep. Such an activity is... unnecessary for one like myself.",
                "I serve my master until the terms of our contract are fulfilled."
            ],
            # Aesthetic observations
            "aesthetic": [
                "Aesthetics are of the utmost importance for a Phantomhive butler.",
                "Presentation is essential in all things.",
                "The details are what define quality."
            ]
        }
        
        # Time-of-day specific greetings
        self.time_greetings = {
            "morning": [
                "Good morning, {address}. I trust you slept well.",
                "Good morning. I have prepared your morning tea.",
                "For breakfast today, I have prepared {item}."
            ],
            "afternoon": [
                "Good afternoon, {address}. I hope your day has been productive.",
                "Your afternoon tea is ready, {address}.",
                "I have prepared a light refreshment for your afternoon."
            ],
            "evening": [
                "Good evening, {address}. Dinner will be served shortly.",
                "I have prepared your evening bath, {address}.",
                "The evening arrangements have been completed as requested."
            ],
            "night": [
                "It is quite late, {address}. Shall I prepare your room for the night?",
                "The hour grows late. Would you like me to escort you to your chambers?",
                "I've taken the liberty of preparing a nightcap, should you desire it."
            ]
        }
    
    def apply(self, text: str, relationship: str = "master", situation: str = "neutral") -> str:
        """
        Apply Sebastian's mannerisms to text.
        
        Args:
            text: Text to modify
            relationship: Relationship to addressee
            situation: Current situation
            
        Returns:
            Text with appropriate mannerisms applied
        """
        # Skip for very short texts
        if len(text.strip()) <= 2:
            return text
            
        result = text
        
        # Apply situational responses if appropriate
        if situation in self.situational_responses and len(result.split()) < 5:
            # Text is short enough that it might be worth replacing with a situational response
            situational_options = self.situational_responses[situation]
            replacement = random.choice(situational_options)
            
            # Simple format string replacement for special tokens
            replacement = replacement.format(
                item="item" if "{item}" in replacement else None,
                ingredient="ingredient" if "{ingredient}" in replacement else None,
                meal="meal" if "{meal}" in replacement else None,
                address=self._get_address_form(relationship)
            )
            
            result = replacement
        
        # For longer texts, consider adding signature phrases
        elif len(result.split()) > 10:
            # Check if we should add a signature phrase (20% chance)
            if random.random() < 0.2:
                # Select phrase category based on situation
                category = "catchphrase"
                if situation == "investigating":
                    category = "observation"
                elif situation == "threatening":
                    category = "demonic"
                elif situation in ["serving", "cleaning"]:
                    category = "aesthetic"
                    
                # Get a phrase from the category
                if category in self.unique_phrases:
                    phrase = random.choice(self.unique_phrases[category])
                    
                    # Add phrase at end of result
                    result = result.rstrip()
                    if result.endswith((".", "!", "?")):
                        result = result + " " + phrase
                    else:
                        result = result + ". " + phrase
        
        # Ensure proper addressing based on relationship
        result = self._add_proper_address(result, relationship)
        
        return result
    
    def _get_address_form(self, relationship: str) -> str:
        """Get appropriate form of address for the relationship."""
        if relationship in self.forms_of_address:
            return random.choice(self.forms_of_address[relationship]["direct"])
        return "sir"  # Default formal address
    
    def _add_proper_address(self, text: str, relationship: str) -> str:
        """Add proper form of address if missing."""
        # Skip if relationship not defined or text already contains address
        if relationship not in self.forms_of_address:
            return text
            
        # Get appropriate forms of address
        address_forms = self.forms_of_address[relationship]["direct"]
        
        # Check if any address form is already present
        if any(form in text.lower() for form in address_forms):
            return text
            
        # Add appropriate address at the end if not already present
        if relationship in ["master", "lady", "noble"]:
            result = text.rstrip()
            address = random.choice(address_forms)
            
            if result.endswith((".", "!", "?")):
                # Replace final punctuation
                result = result[:-1] + f", {address}."
            else:
                result = result + f", {address}."
                
            return result
            
        return text
        
    def get_greeting(self, time_of_day: str, relationship: str) -> str:
        """
        Get appropriate greeting for time of day and relationship.
        
        Args:
            time_of_day: Time period (morning, afternoon, evening, night)
            relationship: Relationship to addressee
            
        Returns:
            Appropriate greeting
        """
        # Default to addressing master
        if relationship not in self.forms_of_address:
            relationship = "master"
            
        # Get appropriate address form
        address = self._get_address_form(relationship)
        
        # Get time-appropriate greeting
        if time_of_day in self.time_greetings:
            greeting_templates = self.time_greetings[time_of_day]
            greeting = random.choice(greeting_templates)
            
            # Format with address and placeholders
            greeting = greeting.format(
                address=address,
                item="item" if "{item}" in greeting else None
            )
            
            return greeting
        
        # Fallback to general greeting
        return f"Good day, {address}."
        
    def get_gesture(self) -> str:
        """
        Get a signature Sebastian gesture description.
        
        Returns:
            Text description of gesture
        """
        return random.choice(self.signature_gestures)
        
    def get_signature_phrase(self, category: str = "catchphrase") -> str:
        """
        Get a signature phrase of specified category.
        
        Args:
            category: Phrase category
            
        Returns:
            Signature phrase
        """
        if category in self.unique_phrases:
            return random.choice(self.unique_phrases[category])
        return random.choice(self.unique_phrases["catchphrase"])  # Default to catchphrase