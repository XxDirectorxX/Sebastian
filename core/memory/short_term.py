"""
Short-term memory system for the Sebastian assistant.
"""
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from core.memory.memory_interface import MemoryEmbeddingInterface

logger = logging.getLogger("Sebastian.ShortTermMemory")

class ShortTermMemory:
    """
    Handles ephemeral, session-based memory that resets frequently.
    Suitable for context windows and recent dialogue tracking.
    """

    def __init__(self, embedding_interface: Optional[MemoryEmbeddingInterface] = None):
        self.memory: List[Dict[str, Any]] = []
        # Use provided embedding interface or create default
        if embedding_interface is None:
            from core.memory.memory_interface import MemoryEmbeddingInterface
            self.embedding_interface = MemoryEmbeddingInterface()
        else:
            self.embedding_interface = embedding_interface

    def remember(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store memory along with computed embedding and metadata.
        """
        embedding = self.embedding_interface.embed(text)
        entry = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
        }
        self.memory.append(entry)
        logger.debug(f"[ShortTermMemory] Memory added: {text}")

    def clear(self) -> None:
        """
        Wipe all stored short-term memory.
        """
        self.memory.clear()
        logger.info("[ShortTermMemory] Memory cleared.")

    def recall(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the last n memories in insertion order.
        """
        return self.memory[-n:] if self.memory else []

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query memories by semantic similarity.
        
        Args:
            query_text: The text to match
            top_k: Maximum number of results to return
            
        Returns:
            List of memory entries ordered by relevance
        """
        if not self.memory:
            return []
            
        corpus = [(entry["text"], entry["embedding"]) for entry in self.memory]
        similar_items = self.embedding_interface.similarity_search(query_text, corpus, top_k)
        
        results = []
        for text, score in similar_items:
            # Find the corresponding memory entry
            matched_entry = next((entry for entry in self.memory if entry["text"] == text), None)
            if matched_entry:
                entry_with_score = matched_entry.copy()
                entry_with_score["similarity_score"] = score
                results.append(entry_with_score)
                
        return results

class InMemoryShortTermStore:
    """
    A basic in-memory short-term store.
    This is not a full implementation of MemoryInterface but provides the logic
    that a MemoryManager could use for its short-term operations.
    """
    def __init__(self):
        # Structure: {session_id: {key: (value, expiry_timestamp_or_None)}}
        self._store: Dict[str, Dict[str, Tuple[Any, Optional[float]]]] = {}
        self._lock = asyncio.Lock() # For thread-safe operations on the store
        logger.info("InMemoryShortTermStore initialized.")

    async def add(self, session_id: str, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        async with self._lock:
            if session_id not in self._store:
                self._store[session_id] = {}
            
            expiry_timestamp = (time.time() + ttl_seconds) if ttl_seconds is not None else None
            self._store[session_id][key] = (value, expiry_timestamp)
            logger.debug(f"Added/Updated short-term memory for session '{session_id}', key '{key}'. TTL: {ttl_seconds}s")

    async def get(self, session_id: str, key: str) -> Optional[Any]:
        async with self._lock:
            session_data = self._store.get(session_id)
            if not session_data:
                return None
            
            item = session_data.get(key)
            if not item:
                return None
            
            value, expiry_timestamp = item
            if expiry_timestamp is not None and time.time() > expiry_timestamp:
                logger.debug(f"Short-term memory expired for session '{session_id}', key '{key}'. Removing.")
                del self._store[session_id][key]
                if not self._store[session_id]: # Clean up empty session
                    del self._store[session_id]
                return None
            return value

    async def get_session_all(self, session_id: str) -> Dict[str, Any]:
        async with self._lock:
            session_data = self._store.get(session_id)
            if not session_data:
                return {}
            
            current_time = time.time()
            active_session_data: Dict[str, Any] = {}
            keys_to_delete = []

            for key, (value, expiry_timestamp) in session_data.items():
                if expiry_timestamp is not None and current_time > expiry_timestamp:
                    keys_to_delete.append(key)
                else:
                    active_session_data[key] = value
            
            if keys_to_delete: # Clean up expired keys within this session
                for key in keys_to_delete:
                    del self._store[session_id][key]
                if not self._store[session_id]: # Clean up empty session
                    del self._store[session_id]
                    logger.debug(f"Cleaned up empty session '{session_id}' after fetching all.")

            return active_session_data

    async def remove(self, session_id: str, key: str) -> bool:
        async with self._lock:
            if session_id in self._store and key in self._store[session_id]:
                del self._store[session_id][key]
                if not self._store[session_id]: # Clean up empty session
                    del self._store[session_id]
                logger.debug(f"Removed short-term memory for session '{session_id}', key '{key}'.")
                return True
            return False

    async def clear_session(self, session_id: str) -> None:
        async with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                logger.debug(f"Cleared all short-term memory for session '{session_id}'.")

    async def cleanup_expired(self) -> None:
        """Actively cleans up all expired items across all sessions."""
        async with self._lock:
            current_time = time.time()
            sessions_to_delete = []
            for session_id, session_data in self._store.items():
                keys_to_delete_in_session = []
                for key, (_, expiry_timestamp) in session_data.items():
                    if expiry_timestamp is not None and current_time > expiry_timestamp:
                        keys_to_delete_in_session.append(key)
                
                for key in keys_to_delete_in_session:
                    del session_data[key]
                
                if not session_data: # If session becomes empty after cleanup
                    sessions_to_delete.append(session_id)
            
            for session_id in sessions_to_delete:
                del self._store[session_id]
            
            if sessions_to_delete or any(keys_to_delete_in_session for _, session_data in self._store.items() for keys_to_delete_in_session in [[]]): # A bit complex check, simplify if needed
                 logger.info(f"Performed active cleanup of expired short-term memories. Removed {len(sessions_to_delete)} empty sessions.")