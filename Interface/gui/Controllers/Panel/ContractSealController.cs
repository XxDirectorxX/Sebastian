using System;

namespace Sebastian.Controllers.Panel
{
    public class ContractSealController
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private readonly QuantumSystemBridge quantumBridge;
        private readonly ParticleSystem particleSystem;
        private float currentPower;
        private float currentIntensity;
        private float currentResonance;
        
        public ContractSealController()
        {
            quantumBridge = new QuantumSystemBridge();
            particleSystem = new ParticleSystem();
            
            InitializeQuantumSystems();
        }
        
        private void InitializeQuantumSystems()
        {
            float fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            quantumBridge.Initialize(fieldEffect);
            particleSystem.Initialize(fieldEffect);
        }
        
        public void ProcessSeal(float intensity)
        {
            currentIntensity = intensity;
            quantumBridge.SynchronizeField(intensity);
            particleSystem.UpdateIntensity(intensity);
        }
        
        public void UpdateSeal(float strength)
        {
            currentPower = strength;
            quantumBridge.UpdateQuantumState(strength);
            particleSystem.UpdateParticles(strength);
        }
        
        public void UpdatePower(float power)
        {
            currentPower = power;
            quantumBridge.UpdateFieldStrength(power);
            particleSystem.UpdatePower(power);
        }
        
        public void UpdateIntensity(float intensity)
        {
            currentIntensity = intensity;
            quantumBridge.UpdateFieldIntensity(intensity);
            particleSystem.UpdateIntensity(intensity);
        }
        
        public void UpdateResonance(float resonance)
        {
            currentResonance = resonance;
            quantumBridge.UpdateFieldResonance(resonance);
            particleSystem.UpdateResonance(resonance);
        }
        
        public void Activate()
        {
            float activationStrength = currentPower * FIELD_STRENGTH;
            quantumBridge.ActivateField(activationStrength);
            particleSystem.Activate();
        }
        
        public void Synchronize()
        {
            float syncCoherence = currentResonance * REALITY_COHERENCE;
            quantumBridge.SynchronizeField(syncCoherence);
            particleSystem.Synchronize();
        }
    }
}
