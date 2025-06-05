import yaml
from pathlib import Path

def test_persona_profile_structure():
    profile_path = Path("assets/personas/persona_profile.yaml")
    assert profile_path.exists()
    with open(profile_path, "r") as f:
        profile = yaml.safe_load(f)
    assert "persona" in profile
    persona = profile["persona"]
    assert "name" in persona
    assert "voice" in persona
    assert "mannerisms" in persona
    assert "error_responses" in persona