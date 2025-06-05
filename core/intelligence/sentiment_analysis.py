"""
Sentiment analysis for Sebastian assistant.
"""

import torch
import asyncio
from typing import Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.labels = ["NEGATIVE", "POSITIVE"]

    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of input text, returning polarity and confidence."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        scores = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
        confidence, label_idx = torch.max(scores, dim=0)
        
        return {
            "label": self.labels[label_idx.item()],
            "score": confidence.item()
        }

    async def analyze_async(self, text: str) -> Dict[str, float]:
        """Asynchronous version of sentiment analysis."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.analyze, text)
        return result

