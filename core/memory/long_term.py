
# core/memory/long_term.py

import os
import json
import logging
from collections import defaultdict
from typing import Dict

logger = logging.getLogger("Sebastian.LongTermMemory")

class LongTermMemory:
    def __init__(self, memory_path: str = "core/memory/data/long_term_memory.json"):
        self.memory_path = memory_path
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        if not os.path.exists(self.memory_path):
            with open(self.memory_path, "w") as f:
                json.dump({}, f)

    def store(self, user_id: str, data: Dict[str, str]):
        full = self._load()
        if user_id not in full:
            full[user_id] = {}
        full[user_id].update(data)
        self._save(full)
        logger.info(f"[LongTermMemory] Stored for {user_id}: {data}")

    def recall(self, user_id: str) -> Dict[str, str]:
        return self._load().get(user_id, {})

    def _load(self):
        with open(self.memory_path, "r") as f:
            return json.load(f)

    def _save(self, data):
        with open(self.memory_path, "w") as f:
            json.dump(data, f, indent=2)
