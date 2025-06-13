
# core/personality/mannerisms.py

from core.personality.persona_manager import get_mannerisms

def apply_mannerism(text: str, tone: str = "neutral") -> str:
    mannerisms = get_mannerisms()
    signature = mannerisms.get("signature", "")

    if tone == "apologetic":
        return mannerisms.get("apology", "") + " " + text
    elif tone == "formal":
        return mannerisms.get("acknowledgement", "") + " " + text
    elif tone == "warm":
        return text + " " + signature
    else:
        return text
