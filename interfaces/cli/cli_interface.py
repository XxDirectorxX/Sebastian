# interfaces/cli/cli_interface.py

import asyncio
from core.intelligence.intent_parser import IntentParser
from core.orchestrator.plugin_orchestrator import PluginOrchestrator
from core.memory.memory_manager import MemoryManager
from core.personality_simulation import PersonalityEngine


class CLIInterface:
    def __init__(self, config):
        self.config = config
        self.intent_parser = IntentParser(config)
        self.memory = MemoryManager(config)
        self.personality = PersonalityEngine(config)
        self.plugins = PluginOrchestrator(config, memory=self.memory, personality=self.personality)

    async def run(self):
        print("╭────────────── CLI Mode ──────────────╮")
        print("│ Type 'exit' to terminate Sebastian. │")
        print("╰──────────────────────────────────────╯")

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() == "exit":
                    print("Sebastian: Until next time, My Lord.")
                    break

                intent = self.intent_parser.parse(user_input)
                response = await self.plugins.dispatch(intent)
                styled = self.personality.apply_tone(response)
                print(f"Sebastian: {styled}")

            except Exception as e:
                print(f"[Error] {e}")
