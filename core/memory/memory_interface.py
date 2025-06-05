"""
Memory interface for Sebastian assistant.

Provides unified access to short-term and long-term memory
systems with asynchronous operations.
"""
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import os
from abc import ABC, abstractmethod

logger = logging.getLogger("Sebastian.MemoryInterface")

class MemoryInterface(ABC):
    """
    Abstract Base Class for all memory systems.
    Defines the contract for short-term and long-term memory operations.
    """

    @abstractmethod
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the memory store, e.g., connect to DB, setup tables."""
        pass

    # --- Short-Term Memory Methods ---
    @abstractmethod
    async def add_short_term_memory(self, session_id: str, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Adds or updates a key-value pair in short-term memory for a given session."""
        pass

    @abstractmethod
    async def get_short_term_memory(self, session_id: str, key: str) -> Optional[Any]:
        """Retrieves a value from short-term memory for a given session and key."""
        pass

    @abstractmethod
    async def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Retrieves all short-term memory for a given session."""
        pass
    
    @abstractmethod
    async def remove_short_term_memory(self, session_id: str, key: str) -> bool:
        """Removes a specific key from short-term memory for a session."""
        pass

    @abstractmethod
    async def clear_short_term_session(self, session_id: str) -> None:
        """Clears all short-term memory for a given session."""
        pass

    # --- Long-Term Memory Methods ---
    @abstractmethod
    async def store_long_term_memory(self, user_id: str, memory_type: str, data: Dict[str, Any], tags: Optional[List[str]] = None) -> str:
        """
        Stores a piece of information in long-term memory.
        Returns a unique ID for the stored memory.
        """
        pass

    @abstractmethod
    async def retrieve_long_term_memories(
        self, 
        user_id: str, 
        query: Optional[str] = None, # For semantic search or keyword search
        memory_type: Optional[str] = None, 
        tags: Optional[List[str]] = None, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieves memories from long-term storage based on criteria.
        The 'query' parameter will be crucial for semantic search later.
        """
        pass

    @abstractmethod
    async def get_long_term_memory_by_id(self, memory_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves a specific long-term memory by its ID."""
        pass
    
    @abstractmethod
    async def update_long_term_memory(self, memory_id: str, data_update: Dict[str, Any], user_id: Optional[str] = None) -> bool:
        """Updates an existing long-term memory."""
        pass

    @abstractmethod
    async def delete_long_term_memory(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        """Deletes a long-term memory by its ID."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Closes connections and releases resources used by the memory system."""
        pass

    # --- Utility/Maintenance Methods (Optional) ---
    async def cleanup_expired_short_term(self) -> None:
        """(Optional) Actively cleans up expired short-term memories if not handled passively."""
        logger.debug("Cleanup of expired short-term memories invoked (if applicable).")
        pass

class MemoryEmbeddingInterface(ABC): # Added MemoryEmbeddingInterface
    """
    Abstract Base Class for memory systems that utilize embeddings for storage and retrieval.
    This interface is typically implemented by long-term memory stores capable of semantic search.
    """

    @abstractmethod
    async def add_memory_with_embedding(
        self, 
        user_id: str, 
        text_content: str, 
        embedding: List[float], # Or appropriate embedding type
        metadata: Dict[str, Any], 
        memory_type: Optional[str] = "generic",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Stores a piece of information along with its pre-computed embedding.
        Returns a unique ID for the stored memory.
        """
        pass

    @abstractmethod
    async def search_memories_by_embedding(
        self, 
        user_id: str, 
        query_embedding: List[float], # Or appropriate embedding type
        limit: int = 5,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        # Additional filters like time range could be added
    ) -> List[Dict[str, Any]]: # Each dict could include 'payload', 'score', 'id'
        """
        Retrieves memories that are semantically similar to the query embedding.
        """
        pass
    
    # This interface might also re-declare or assume some methods from MemoryInterface
    # if it's meant to be a complete memory solution, e.g., initialize, close.
    # For now, focusing on the embedding-specific methods.
    # If a class implements both, it will inherit methods from both.
    # Alternatively, MemoryEmbeddingInterface could inherit from MemoryInterface:
    # class MemoryEmbeddingInterface(MemoryInterface):
    # This would mean any class implementing MemoryEmbeddingInterface must also implement all of MemoryInterface.
    # Let's keep it separate for now to allow more flexibility, a store might ONLY do embeddings.
