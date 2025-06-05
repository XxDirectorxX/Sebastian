"""
Long-term memory system for the Sebastian assistant.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

LONG_TERM_MEMORY_FILE = "core/memory/data/long_term_memory.json"

class LongTermMemory:
    """Stores enduring facts, preferences, and relationships over long durations."""

    def __init__(self, embedding_interface=None):
        """Initialize long-term memory with persistence and optional embedding interface."""
        self.memory_file = LONG_TERM_MEMORY_FILE
        
        # Lazy import to avoid circular dependencies
        if embedding_interface is None:
            from core.memory.memory_interface import MemoryEmbeddingInterface
            self.embedding_interface = MemoryEmbeddingInterface()
        else:
            self.embedding_interface = embedding_interface
            
        self.entries: List[Dict[str, Any]] = self._load_memory()

    def _load_memory(self) -> List[Dict[str, Any]]:
        """Load memory from persistent storage."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse memory file: {e}")
                return []
        return []

    def _save_memory(self):
        """Save memory to persistent storage."""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.entries, f, indent=2)
        logger.debug("Memory saved to disk")

    def remember(self, text: str, source: str = "system", tags: Optional[List[str]] = None):
        """Add a new fact or knowledge entry to long-term memory."""
        embedding = self.embedding_interface.embed(text)
        entry = {
            "text": text,
            "source": source,
            "tags": tags or [],
            "timestamp": datetime.utcnow().isoformat(),
            "embedding": embedding
        }
        self.entries.append(entry)
        self._save_memory()
        logger.debug(f"[LongTermMemory] Remembered new fact: {text}")

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top_k most relevant facts based on semantic similarity to the query."""
        if not self.entries:
            return []

        corpus = [(entry["text"], entry["embedding"]) for entry in self.entries]
        similar_items = self.embedding_interface.similarity_search(query_text, corpus, top_k)

        results = []
        for text, score in similar_items:
            # Find the corresponding memory entry
            matched_entry = next((entry for entry in self.entries if entry["text"] == text), None)
            if matched_entry:
                entry_with_score = matched_entry.copy()
                entry_with_score["similarity_score"] = score
                results.append(entry_with_score)

        return results

    def delete(self, text: str):
        """Delete an entry matching the given text from long-term memory."""
        original_len = len(self.entries)
        self.entries = [e for e in self.entries if e["text"] != text]
        if len(self.entries) < original_len:
            self._save_memory()
            logger.info(f"Deleted entry from long-term memory: {text}")
        else:
            logger.warning(f"No matching entry found to delete: {text}")
