# core/dialogue_flow.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from core.context_manager import ContextManager

class DialogueFlow:
    def __init__(self, model_dir: str = None):
        """
        Initialize DialogueFlow with sentiment analysis capabilities.

        Args:
            model_dir (str): Path to the sentiment model directory.
        """
        self.context_manager = ContextManager()
        if model_dir is None:
            # Default path for sentiment model, adjust if needed
            model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "sentiment")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def analyze_sentiment(self, text: str):
        """
        Analyze sentiment of the input text.

        Args:
            text (str): The user input text.

        Returns:
            float: Probability of positive sentiment (range 0 to 1).
            int: Predicted sentiment label index.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            positive_prob = probabilities[0][1].item()
            predicted_label = torch.argmax(probabilities, dim=1).item()
        return positive_prob, predicted_label

    def process_input(self, user_input: str):
        """
        Process user input, analyze sentiment, update context, and return sentiment info.

        Args:
            user_input (str): Text input from user.

        Returns:
            dict: Contains 'sentiment_probability', 'sentiment_label', and updated context.
        """
        pos_prob, label = self.analyze_sentiment(user_input)
        sentiment_label = "positive" if label == 1 else "negative"

        # Update conversational context with sentiment info
        self.context_manager.update_context("last_sentiment", {
            "probability": pos_prob,
            "label": sentiment_label,
        })

        # Potentially extend context updates, dialogue history, etc.
        self.context_manager.update_context("last_user_input", user_input)

        return {
            "sentiment_probability": pos_prob,
            "sentiment_label": sentiment_label,
            "context": self.context_manager.get_context(),
        }


