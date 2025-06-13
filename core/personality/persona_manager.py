
# core/personality/persona_manager.py

import yaml
import logging
from pathlib import Path

logger = logging.getLogger("Sebastian.PersonaManager")

class PersonaManager:
    def __init__(self, config_path: str):
        self.profile = self._load_profile(config_path)

    def _load_profile(self, path: str) -> dict:
        try:
            with open(Path(path), "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.exception(f"Failed to load persona profile from {path}")
            return {}

    def get_mannerism(self, key: str) -> str:
        return self.profile.get("persona", {}).get("mannerisms", {}).get(key, "")

    def get_error_response(self, key: str) -> str:
        return self.profile.get("persona", {}).get("error_responses", {}).get(key, "")

    def get_signature_phrase(self) -> str:
        phrases = self.profile.get("signature_phrases", [])
        return phrases[0] if phrases else "I am, after all, simply one hell of a butler."

    def get_context_modifiers(self) -> dict:
        return self.profile.get("persona", {}).get("context_modifiers", {})

    def get_tone_rules(self) -> dict:
        return self.profile.get("tone_rules", {})
