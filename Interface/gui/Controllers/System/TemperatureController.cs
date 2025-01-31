using System;

namespace Sebastian.Controllers.System
{
    public class TemperatureController
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private readonly QuantumSystemBridge quantumBridge;
        private float currentTemperature;
        private string currentMode;
        
        public TemperatureController()
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
        
        public void SetTemperature(float temperature)
        {
            currentTemperature = temperature;
            UpdateTemperatureState();
        }
        
        public void SetMode(string mode)
        {
            currentMode = mode;
            UpdateTemperatureState();
        }
        
        public float GetCurrentIntensity()
        {
            return currentTemperature * REALITY_COHERENCE;
        }
        
        private void UpdateTemperatureState()
        {
            float fieldStrength = currentTemperature * FIELD_STRENGTH;
            float coherence = GetModeCoherence() * REALITY_COHERENCE;
            
            quantumBridge.UpdateFieldStrength(fieldStrength);
            quantumBridge.UpdateFieldCoherence(coherence);
        }
        
        private float GetModeCoherence()
        {
            return currentMode switch
            {
                "Heat" => 0.8f,
                "Cool" => 0.6f,
                _ => 1.0f
            };
        }
    }
}
