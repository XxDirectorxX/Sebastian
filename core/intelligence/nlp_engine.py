"""
NLP Engine for Sebastian assistant.
"""

import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Dict, Any, Optional
from core.embeddings import EmbeddingEngine # Assuming this import is correct
import logging
import asyncio

logger = logging.getLogger("Sebastian.NLPEngine")


class NLPModel:
    """
    NLPModel provides semantic embedding capabilities using a transformer-based encoder.
    It is responsible for converting raw text into vector representations that can be used
    for intent recognition, memory recall, context tracking, and downstream AI reasoning.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the NLP model with tokenizer and transformer encoder.

        Args:
            model_name (str): HuggingFace model identifier.
        """
        # Use the centralized embedding engine
        self.embedding_engine = EmbeddingEngine(model_name)

    def get_embeddings(self, text: Union[str, List[str]]) -> Tensor:
        """Convert text input to vector embeddings.

        Args:
            text (Union[str, List[str]]): String or list of strings to embed.

        Returns:
            Tensor: Embedding vectors.
        """
        if isinstance(text, str):
            # Ensure the output of embed/batch_embed is suitable for torch.tensor()
            # If they already return tensors, the torch.tensor() call might be redundant
            # or could be used for copying/type conversion.
            # Assuming embed/batch_embed return list of floats or numpy arrays.
            return torch.tensor(self.embedding_engine.embed(text))
        else:
            return torch.tensor(self.embedding_engine.batch_embed(text))

    def get_embedding_dim(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            int: Embedding dimension
        """
        return self.embedding_engine.get_embedding_dim()

    def is_cuda(self) -> bool:
        """
        Check if model is using CUDA.

        Returns:
            bool: True if using CUDA, False otherwise
        """
        # This checks global PyTorch CUDA availability, not specific model placement.
        # The EmbeddingEngine should ideally handle its own device placement.
        return torch.cuda.is_available()


# For a more structured approach, consider a dataclass or Pydantic model later
# from dataclasses import dataclass
# @dataclass
# class ParsedIntent:
#     intent_name: str
#     entities: Dict[str, Any]
#     confidence: float = 1.0
#     original_text: str = ""

class NLPEngine:
    """
    Natural Language Processing Engine for Sebastian.
    Responsible for understanding user input, parsing intent, and extracting entities.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config else {}
        self.language = self.config.get("language", "en")
        # Placeholder for future model loading, e.g., Hugging Face transformers
        # self.tokenizer = None
        # self.model = None
        logger.info(f"NLPEngine initialized for language: {self.language}.")
        # self._load_models() # Method to load actual NLP models

    def _load_models(self):
        """
        (Future) Placeholder for loading NLP models (e.g., from Hugging Face, spaCy).
        This would be an asynchronous method if model loading is I/O bound.
        """
        logger.info("NLP models would be loaded here.")
        # Example:
        # from transformers import pipeline
        # self.intent_classifier = pipeline("text-classification", model="your_intent_model")
        # self.ner_tagger = pipeline("ner", model="your_ner_model")
        pass

    async def parse(self, text: str) -> Dict[str, Any]:
        """
        Parses the input text to determine intent and extract entities.
        This is a basic placeholder implementation.
        
        Returns:
            A dictionary representing the parsed intent, e.g.,
            {
                "intent_name": "get_weather",
                "entities": {"location": "London", "date": "today"},
                "confidence": 0.85,
                "original_text": text
            }
        """
        logger.debug(f"Parsing text: '{text}'")
        # Simulate async processing
        await asyncio.sleep(0.01) # Placeholder for actual NLP processing time

        text_lower = text.lower().strip()
        intent_name: str = "unknown"
        entities: Dict[str, Any] = {}
        confidence: float = 0.5 # Default low confidence for basic matching

        # --- Basic Keyword-Based Intent Recognition (Placeholder) ---
        # Accessing intents from config if available
        configured_intents = self.config.get("intents", {})

        # More robust matching using configured intents
        for intent, keywords in configured_intents.items():
            if any(keyword in text_lower for keyword in keywords):
                intent_name = intent
                confidence = 0.7 # Default for keyword match, can be refined
                # Basic entity extraction can be added here based on intent
                if intent == "get_weather":
                    # Example: look for "in [location]" or "for [location]"
                    if " in " in text_lower:
                        try:
                            entities["location"] = text_lower.split(" in ", 1)[1].split("?")[0].strip()
                        except IndexError:
                            pass
                    elif " for " in text_lower:
                        try:
                            entities["location"] = text_lower.split(" for ", 1)[1].split("?")[0].strip()
                        except IndexError:
                            pass
                elif intent == "set_user_name":
                    if "my name is " in text_lower:
                        try:
                            entities["user_name"] = text_lower.split("my name is ", 1)[1].strip().split(" ")[0]
                        except IndexError:
                            pass
                    elif "call me " in text_lower:
                         try:
                            entities["user_name"] = text_lower.split("call me ", 1)[1].strip().split(" ")[0]
                         except IndexError:
                            pass
                elif intent == "create_memory":
                    if "remember that " in text_lower:
                        try:
                            entities["content"] = text_lower.split("remember that ", 1)[1].strip()
                        except IndexError:
                            pass
                    elif "make a note " in text_lower: # Corrected "take a note" to "make a note" if that's the keyword
                        try:
                            entities["content"] = text_lower.split("make a note ", 1)[1].strip()
                        except IndexError:
                            pass
                # Add more intent-specific entity extraction here
                break # Stop after first match for simplicity; could be improved

        # Fallback if no configured intent matched but old hardcoded rules might apply
        if intent_name == "unknown":
            if "hello" in text_lower or "hi" in text_lower or "greetings" in text_lower:
                intent_name = "greet"
                confidence = 0.9
            elif "lights" in text_lower and ("turn on" in text_lower or "activate" in text_lower):
                intent_name = "control_device"
                entities["device_name"] = "lights"
                entities["action"] = "on"
                confidence = 0.75
            elif "lights" in text_lower and ("turn off" in text_lower or "deactivate" in text_lower):
                intent_name = "control_device"
                entities["device_name"] = "lights"
                entities["action"] = "off"
                confidence = 0.75


        parsed_result = {
            "intent_name": intent_name,
            "entities": entities,
            "confidence": confidence,
            "original_text": text
        }
        
        logger.info(f"Parsed result: {parsed_result}")
        return parsed_result

    async def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        (Future) Analyzes the sentiment of the given text.
        Returns a dictionary with sentiment scores (e.g., positive, negative, neutral, compound).
        """
        # Placeholder - to be implemented using libraries like VADER, TextBlob, or transformer models
        logger.debug(f"Sentiment analysis requested for: '{text}' (Not yet implemented)")
        await asyncio.sleep(0.01)
        # Example output structure
        return {"polarity": 0.0, "subjectivity": 0.0, "compound": 0.0} # Neutral default

    async def get_dialogue_response(self, current_intent: Dict[str, Any], conversation_history: Optional[list] = None) -> str:
        """
        (Future) Manages dialogue flow and generates a contextual response.
        This would interact with a dialogue management system or rules.
        """
        logger.debug(f"Dialogue response requested for intent: {current_intent} (Not yet implemented)")
        await asyncio.sleep(0.01)
        return "I am processing your request." # Placeholder

# Example usage (for testing purposes, not part of the class)
async def _test_nlp_engine():
    # Example config for testing the parser with configured intents
    test_config = {
        "language": "en",
        "intents": {
            "greet": ["hello", "hi", "greetings"],
            "get_weather": ["weather", "forecast"],
            "get_time": ["time", "what time is it"],
            "create_memory": ["remember that", "make a note"],
            "set_user_name": ["my name is", "call me"]
        }
    }
    nlp = NLPEngine(config=test_config.get("nlp_engine", test_config)) # Pass the nlp_engine part or the whole config
    test_phrases = [
        "Hello Sebastian",
        "What's the weather like in London?",
        "Turn on the lights", # This will use fallback if not in configured_intents
        "Remember that I prefer tea at 4 PM",
        "My name is Michael",
        "This is gibberish"
    ]
    for phrase in test_phrases:
        result = await nlp.parse(phrase)
        print(f"Input: '{phrase}' -> Parsed: {result}")

if __name__ == "__main__":
    # Configure basic logging for standalone testing
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # asyncio.run(_test_nlp_engine()) # Uncomment to run test
    pass
