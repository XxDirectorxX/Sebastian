# core/context_manager.py

"""
context_manager.py

Maintains conversation context, memory, and state for ongoing dialogues.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ContextManager:
    """
    Maintains conversation state and manages memory across dialogue turns.
    """
    
    def __init__(self):
        self._context = {}
        logger.info("Context manager initialized")
        
    def update_context(self, key: str, value: Any) -> None:
        """
        Update a specific context value.
        
        Args:
            key: The context key to update
            value: The new value
        """
        self._context[key] = value
        logger.debug(f"Updated context key '{key}'")
        
    def get_context(self, key: Optional[str] = None) -> Any:
        """
        Retrieve context information.
        
        Args:
            key: Specific context key to retrieve, or None for entire context
            
        Returns:
            The requested context value or the entire context dictionary
        """
        if key is None:
            return self._context
        return self._context.get(key)
        
    def clear_context(self) -> None:
        """Reset the entire context."""
        self._context = {}
        logger.info("Context cleared")
