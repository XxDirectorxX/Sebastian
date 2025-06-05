"""
Intent parser for Sebastian assistant — upgraded to semantic ranking and plugin-aligned parsing.
"""
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from core.intelligence.nlp_engine import NLPModel
from core.intelligence.ner_engine import NEREngine
from core.plugins import __all__ as registered_plugins
from core.plugin_utils import get_plugin_examples

logger = logging.getLogger(__name__)

class IntentParser:
    """
    SOTA hybrid intent parser with semantic ranking, NER heuristics,
    and alignment to plugin metadata.
    """

    def __init__(self,
                 semantic_model: Optional[NLPModel] = None,
                 ner_model: Optional[NEREngine] = None):
        self.semantic_model = semantic_model or NLPModel()
        self.ner_model = ner_model or NEREngine()
        logger.info("[IntentParser] Initialized with semantic and NER engines.")

        # Precompute plugin intent embeddings
        self.plugin_intent_examples = self._load_plugin_examples()
        self.plugin_embeddings = self._compute_plugin_embeddings(self.plugin_intent_examples)

    def _load_plugin_examples(self) -> Dict[str, List[str]]:
        examples = {}
        for plugin in registered_plugins:
            phrases = get_plugin_examples(plugin)
            if phrases:
                examples[plugin] = phrases
        return examples

    def _compute_plugin_embeddings(self, plugin_examples: Dict[str, List[str]]) -> Dict[str, List[np.ndarray]]:
        return {
            plugin: [self.semantic_model.get_embedding(phrase) for phrase in phrases]
            for plugin, phrases in plugin_examples.items()
        }

    def parse(self, text: str) -> Dict[str, Any]:
        logger.debug("Parsing input: %s", text)

        # Semantic embedding of user text
        input_embedding = self.semantic_model.get_embedding(text)

        # Entity extraction
        entities = self.ner_model.extract_entities(text)
        people = self.ner_model.get_named_people(text)
        noun_phrases = self.ner_model.get_noun_chunks(text)

        # Infer top matching plugin intent
        intent, confidence = self._infer_intent(text, input_embedding)

        return {
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
            "people": people,
            "noun_phrases": noun_phrases,
            "embedding": input_embedding,
            "raw_text": text
        }

    def _infer_intent(self, text: str, input_embedding: np.ndarray) -> Tuple[str, float]:
        best_plugin = None
        best_score = -1.0

        for plugin, embeddings in self.plugin_embeddings.items():
            for ref_embedding in embeddings:
                score = self._cosine_similarity(input_embedding, ref_embedding)
                if score > best_score:
                    best_score = score
                    best_plugin = plugin

        threshold = 0.75  # Confidence threshold for intent acceptance
        if best_score >= threshold:
            logger.debug("Intent '%s' matched with confidence %.2f", best_plugin, best_score)
            return best_plugin, best_score
        else:
            logger.debug("No strong intent match found (max score = %.2f)", best_score)
            return "unknown", best_score

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        if vec1 is None or vec2 is None:
            return -1.0
        dot = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot / (norm_a * norm_b + 1e-9)
