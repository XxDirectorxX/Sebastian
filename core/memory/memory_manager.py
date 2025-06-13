
# core/memory/memory_manager.py

import logging
from datetime import datetime
from core.memory.short_term import ShortTermMemory
from core.memory.long_term import LongTermMemory
from core.memory.vector_store import get_collection
from core.memory.embeddings import Embedder

logger = logging.getLogger("Sebastian.MemoryManager")

class MemoryManager:
    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        self.embedder = Embedder()
        self.collection = get_collection()

    def store(self, user_id: str, text: str):
        timestamp = datetime.utcnow().isoformat()
        embedding = self.embedder.embed(text)
        self.short_term.store(user_id, text)
        self.long_term.store(user_id, {timestamp: text})
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[f"{user_id}_{timestamp}"],
            metadatas=[{"user": user_id, "timestamp": timestamp}]
        )
        logger.info(f"[MemoryManager] Stored memory: {text}")

    def recall_recent(self, user_id: str, limit: int = 5):
        return self.short_term.recall(user_id, limit=limit)

    def search_semantic(self, query: str, top_k: int = 3):
        embedding = self.embedder.embed(query)
        return self.collection.query(query_embeddings=[embedding], n_results=top_k)
