
# core/memory/short_term.py

import time
from typing import List, Dict, Any
import logging

logger = logging.getLogger("Sebastian.ShortTermMemory")

class ShortTermMemory:
    def __init__(self, ttl_seconds: int = 600):
        self.ttl = ttl_seconds
        self.memory: List[Dict[str, Any]] = []

    def store(self, user_id: str, content: str):
        timestamp = time.time()
        self.memory.append({
            "user_id": user_id,
            "content": content,
            "timestamp": timestamp
        })
        logger.info(f"[ShortTermMemory] Stored for {user_id}: {content}")

    def recall(self, user_id: str, limit: int = 5) -> List[str]:
        now = time.time()
        self.memory = [m for m in self.memory if now - m["timestamp"] <= self.ttl]
        user_entries = [m["content"] for m in self.memory if m["user_id"] == user_id]
        return user_entries[-limit:]
