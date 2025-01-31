using System;

namespace Sebastian.Controllers.Panel
{
    public class EnergyProjectionController
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private readonly QuantumSystemBridge quantumBridge;
        private readonly ParticleSystem particleSystem;
        private float currentIntensity;
        private float currentFocus;
        private float currentWaveform;
        
        public EnergyProjectionController()
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
        
        public void ProcessEnergy(float intensity)
        {
            currentIntensity = intensity;
            quantumBridge.SynchronizeField(intensity);
            particleSystem.UpdateIntensity(intensity);
        }
        
        public void UpdateEnergy(float strength)
        {
            quantumBridge.UpdateQuantumState(strength);
            particleSystem.UpdateParticles(strength);
        }
        
        public void UpdateIntensity(float intensity)
        {
            currentIntensity = intensity;
            quantumBridge.UpdateFieldIntensity(intensity);
            particleSystem.UpdateIntensity(intensity);
        }
        
        public void UpdateFocus(float focus)
        {
            currentFocus = focus;
            quantumBridge.UpdateFieldFocus(focus);
            particleSystem.UpdateFocus(focus);
        }
        
        public void UpdateWaveform(float waveform)
        {
            currentWaveform = waveform;
            quantumBridge.UpdateFieldWaveform(waveform);
            particleSystem.UpdateWaveform(waveform);
        }
        
        public void Project()
        {
            float projectionStrength = currentIntensity * FIELD_STRENGTH;
            quantumBridge.ProjectField(projectionStrength);
            particleSystem.Project();
        }
        
        public void Stabilize()
        {
            float stabilityCoherence = currentWaveform * REALITY_COHERENCE;
            quantumBridge.StabilizeField(stabilityCoherence);
            particleSystem.Stabilize();
        }
    }
}
