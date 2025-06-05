# core/tts_voice_router.py

class VoiceRouter:
    def __init__(self):
        # Predefined voice profiles mapped by personality, mood, or user choice
        self.voice_profiles = {
            "sebastian_default": {
                "voice_id": None,  # Default system voice
                "rate": 140,
                "volume": 1.0
            },
            "sebastian_intense": {
                "voice_id": None,
                "rate": 160,
                "volume": 1.0
            },
            # Add more voice profiles here as needed
        }
        self.current_profile = "sebastian_default"

    def set_profile(self, profile_name):
        if profile_name in self.voice_profiles:
            self.current_profile = profile_name
            return True
        return False

    def get_current_profile(self):
        return self.voice_profiles[self.current_profile]
