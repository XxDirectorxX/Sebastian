"""
Embedding utilities for semantic representation of text across all system modules.
"""
import torch
import numpy as np
import logging
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple, Union, Dict, Any

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """
    Provides unified vector embeddings for text using transformer models.
    Used by both memory and intelligence subsystems for consistent semantic representations.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with specified model or default to efficient all-MiniLM."""
        try:
            self.model = SentenceTransformer(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logger.info(f"EmbeddingEngine initialized with model {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        with torch.no_grad():
            embedding = self.model.encode(text, convert_to_tensor=True).cpu().numpy().tolist()
        return embedding
    
    def batch_embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts at once."""
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True).cpu().numpy()
        return embeddings
        
    def similarity_search(self, query: str, corpus: List[Tuple[str, List[float]]], 
                         top_k: int = 3) -> List[Tuple[str, float]]:
        """Find most similar items to query text."""
        query_embedding = np.array(self.embed(query))
        scores = []
        
        for text, embed in corpus:
            sim_score = util.cos_sim(query_embedding, np.array(embed)).item()
            scores.append((text, sim_score))
            
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    
    def get_embedding_dim(self) -> int:
        """Returns the dimension of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()