using System;

namespace Sebastian.Controllers.System
{
    public class LightingController
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private readonly QuantumSystemBridge quantumBridge;
        private string currentRoom;
        private float currentBrightness;
        private float currentColorTemp;
        
        public LightingController()
        {
            quantumBridge = new QuantumSystemBridge();
            InitializeQuantumSystems();
        }
        
        private void InitializeQuantumSystems()
        {
            float fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            quantumBridge.Initialize(fieldEffect);
        }
        
        public void Initialize(float fieldStrength)
        {
            quantumBridge.UpdateQuantumState(fieldStrength);
        }
        
        public void SelectRoom(string room)
        {
            currentRoom = room;
            UpdateLightingState();
        }
        
        public void SetBrightness(float brightness)
        {
            currentBrightness = brightness;
            UpdateLightingState();
        }
        
        public void SetColorTemperature(float colorTemp)
        {
            currentColorTemp = colorTemp;
            UpdateLightingState();
        }
        
        public float GetCurrentIntensity()
        {
            return currentBrightness * REALITY_COHERENCE;
        }
        
        private void UpdateLightingState()
        {
            float fieldStrength = currentBrightness * FIELD_STRENGTH;
            float coherence = currentColorTemp * REALITY_COHERENCE;
            
            quantumBridge.UpdateFieldStrength(fieldStrength);
            quantumBridge.UpdateFieldCoherence(coherence);
        }
    }
}
