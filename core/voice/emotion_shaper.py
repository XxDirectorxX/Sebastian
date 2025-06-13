
# core/voice/emotion_shaper.py

import logging

logger = logging.getLogger("Sebastian.EmotionShaper")

EMOTION_PROFILES = {
    "neutral":    {"pitch": 1.0, "speed": 1.0, "energy": 1.0},
    "warm":       {"pitch": 1.05, "speed": 0.95, "energy": 1.1},
    "serious":    {"pitch": 0.95, "speed": 0.98, "energy": 0.9},
    "sarcastic":  {"pitch": 1.1, "speed": 1.05, "energy": 0.85},
    "urgent":     {"pitch": 1.2, "speed": 1.2, "energy": 1.3},
    "apologetic": {"pitch": 0.9, "speed": 0.9, "energy": 0.7},
    "enthusiastic": {"pitch": 1.15, "speed": 1.1, "energy": 1.2}
}

def get_emotion_config(emotion: str = "neutral") -> dict:
    config = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["neutral"])
    logger.info(f"[EmotionShaper] Applying profile for emotion: {emotion} -> {config}")
    return config
