import torch
import numpy as np
import pygame
import sounddevice as sd
from gtts import gTTS
from io import BytesIO
from datetime import datetime
from Core.quantum_unified_core import QuantumUnifiedCore

print("SEBASTIAN QUANTUM AI SYSTEM")
print("===========================")

class SebastianCore:
    def __init__(self):
        self.field_strength = 46.97871376
        self.reality_coherence = 1.618033988749895
        self.quantum_core = QuantumUnifiedCore()
        pygame.mixer.init()
    
    def activate_quantum_field(self):
        print("\nActivating Quantum Field...")
        print(f"Field Strength: {self.field_strength}")
        print(f"Reality Coherence: {self.reality_coherence}")
        self.quantum_core.execute_unified(self.field_strength)
        print("\nAll Systems Online!")
        print("Ready for quantum operations.")
        
    def speak(self, text):
        print(f"Sebastian: {text}")
        tts = gTTS(text=text, lang='en')
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
    
    def listen(self):
        print("\nAwaiting your command, my lord...")
        try:
            duration = 3
            recording = sd.rec(int(duration * 44100), 
                             samplerate=44100, 
                             channels=1, 
                             blocking=True)
            self.speak("I shall execute your command with perfect precision, my lord.")
            
        except KeyboardInterrupt:
            print("\nVoice System Deactivated")

if __name__ == "__main__":
    sebastian = SebastianCore()
    sebastian.activate_quantum_field()
    while True:
        sebastian.listen()