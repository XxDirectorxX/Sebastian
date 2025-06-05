# core/intelligence/ner_engine.py

"""
Named Entity Recognition engine for Sebastian assistant.
"""
import spacy
from typing import Dict, List

class NEREngine:
    """
    Lightweight NLP engine for named entity recognition and syntactic parsing using spaCy.
    Complements the transformer-based NLPModel with structural linguistic capabilities.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = spacy.load(model_name)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities (e.g., time, date, people, places) from text.

        Args:
            text (str): User input.

        Returns:
            Dict[str, List[str]]: Entities grouped by label.
        """
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
            
        return entities

    def get_named_people(self, text: str) -> List[str]:
        """
        Extract people's names from text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of identified person names
        """
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    def get_noun_chunks(self, text: str) -> List[str]:
        """
        Extract noun phrases from text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of noun phrases
        """
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]
