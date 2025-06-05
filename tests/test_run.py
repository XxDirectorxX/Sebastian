import pytest
from main import create_default_config, validate_configuration

def test_create_default_config_structure():
    config = create_default_config()
    assert "voice" in config
    assert "memory" in config
    assert "personality" in config

def test_validate_configuration_missing_sections(monkeypatch):
    config = {}
    validate_configuration(config)
    assert "voice" in config
    assert "memory" in config
    assert "personality" in config