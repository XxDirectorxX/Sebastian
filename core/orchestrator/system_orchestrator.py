# core/orchestrator/system_orchestrator.py

import asyncio
from core.orchestrator.plugin_orchestrator import PluginOrchestrator
from core.voice.voice_interface import VoiceInterface
from core.memory.memory_manager import MemoryManager
from core.personality_simulation import PersonalityEngine
from core.intelligence.intent_parser import IntentParser


class SystemOrchestrator:
    def __init__(self, config: dict):
        self.config = config

        self.memory = MemoryManager(config)
        self.personality = PersonalityEngine(config)
        self.plugins = PluginOrchestrator(config, memory=self.memory, personality=self.personality)
        self.voice = VoiceInterface(config)
        self.intent_parser = IntentParser(config)

    async def run(self):
        print("[Orchestrator] Starting Sebastian daemon.")
        await self.voice.initialize()

        while True:
            try:
                transcript = await self.voice.listen_for_command()
                if not transcript:
                    continue

                print(f"[User] {transcript}")
                intent = self.intent_parser.parse(transcript)
                response = await self.plugins.dispatch(intent)

                if response:
                    spoken = self.personality.apply_tone(response)
                    await self.voice.speak(spoken)

            except asyncio.CancelledError:
                print("[Orchestrator] Cancelled.")
                break
            except Exception as e:
                print(f"[Orchestrator] Error: {e}")
