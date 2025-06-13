# core/personality_simulation.py

import yaml
import os
import re
import random

class PersonalityEngine:
    def __init__(self, config):
        profile_path = config['personality']['profile']
        with open(profile_path, 'r', encoding='utf-8') as f:
            self.profile = yaml.safe_load(f)['persona']

        self.rules = self.profile.get("behavior_rules", {})
        self.tones = self.profile.get("emotional_tone", {})
        self.manner = self.profile.get("mannerisms", {})
        self.errors = self.profile.get("error_responses", {})

    def apply_tone(self, response_text: str, emotion: str = "neutral") -> str:
        if not response_text:
            return self.errors.get("generic", "My apologies, My Lord. Something went awry.")

        response_text = self._diction_rewrite(response_text)

        if self.rules.get("sarcasm_threshold", 0) >= 0.7 and emotion == "sarcastic":
            response_text = self._sarcasm_wrap(response_text)

        if self.rules.get("verbosity", "concise") == "concise":
            response_text = self._trim_response(response_text)

        acknowledgement = self.manner.get("acknowledgement", "Very well")
        signature = self.manner.get("signature", "I am, after all, simply one hell of a butler.")

        return f"{acknowledgement}. {response_text.strip().capitalize()}. {signature}"

    def get_error_response(self, code: str) -> str:
        return self.errors.get(code, self.errors.get("generic"))

    def _sarcasm_wrap(self, text):
        candidates = [
            f"Oh, absolutely. {text.lower()}... just as we expected.",
            f"Indeed. {text.lower()}, how original.",
            f"Of course. {text.lower()} — what could possibly go wrong?"
        ]
        return random.choice(candidates)

    def _trim_response(self, text):
        if len(text) > 120:
            return text[:117] + "..."
        return text

    def _diction_rewrite(self, text):
        replacements = {
            r"\bcan't\b": "cannot",
            r"\bwon't\b": "will not",
            r"\bdon't\b": "do not",
            r"\bokay\b": "certainly",
            r"\bhi\b": "greetings",
            r"\byeah\b": "indeed",
            r"\byou\b": "you, My Lord",
            r"\bthanks\b": "my gratitude",
            r"\bbye\b": "farewell",
            r"\buh\b": "",
            r"\bum\b": "",
            r"\blike\b": "",
            r"\bwhatever\b": "as you prefer",
            r"\bno problem\b": "it is my duty",
            r"\bsure\b": "but of course",
            r"\bhello\b": "good day",
            r"\bwhat's up\b": "how may I serve you, My Lord"
        }
        for pattern, repl in replacements.items():
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        return text
