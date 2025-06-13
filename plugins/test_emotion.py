# plugins/test_emotion.py

def test_emotion():
    """
    Return a test string to be modified by the PersonalityEngine
    using different emotional tones.
    """
    return "I understand the request and will act accordingly"

# Register this under multiple mock intents
test_emotion.intent = "test_emotion"
