import pytest
import asyncio
from core.orchestrator.system_integrator import SystemIntegrator

@pytest.mark.asyncio
async def test_system_integrator_initialization():
    config = {
        "voice": {"enabled": True},
        "vision": {"enabled": True},
        "memory": {}
    }
    persona_say = lambda msg, error=False: None
    integrator = SystemIntegrator(config, persona_say)
    await integrator.initialize()
    health = integrator.health_check()
    assert health["voice"]
    assert health["vision"]
    assert health["memory"]
    assert health["initialized"]

@pytest.mark.asyncio
async def test_system_integrator_shutdown():
    config = {
        "voice": {"enabled": True},
        "vision": {"enabled": True},
        "memory": {}
    }
    persona_say = lambda msg, error=False: None
    integrator = SystemIntegrator(config, persona_say)
    await integrator.initialize()
    await integrator.shutdown()