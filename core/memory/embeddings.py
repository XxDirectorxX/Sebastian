"""
Embedding utilities for semantic representation of text.
"""
import numpy as np
from core.embeddings import EmbeddingEngine
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple

class EmbeddingEngine:
    """Provides vector embeddings for text using transformer models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True).tolist()
        
    def similarity_search(self, query: str, corpus: List[Tuple[str, List[float]]], 
                         top_k: int = 3) -> List[Tuple[str, float]]:
        """Find most similar items to query."""
        query_embedding = np.array(self.embed(query))
        scores = [(text, util.cos_sim(query_embedding, np.array(embed)).item()) 
                 for text, embed in corpus]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]