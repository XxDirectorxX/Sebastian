using System;

namespace Sebastian.Controllers.Panel
{
    public class TimeControlController
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private readonly QuantumSystemBridge quantumBridge;
        private readonly ParticleSystem particleSystem;
        private float currentFlow;
        private float currentDilation;
        private float currentSync;
        
        public TimeControlController()
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
        
        public void ProcessTime(float intensity)
        {
            currentFlow = intensity;
            quantumBridge.SynchronizeField(intensity);
            particleSystem.UpdateIntensity(intensity);
        }
        
        public void UpdateTime(float strength)
        {
            quantumBridge.UpdateQuantumState(strength);
            particleSystem.UpdateParticles(strength);
        }
        
        public void UpdateFlow(float flow)
        {
            currentFlow = flow;
            quantumBridge.UpdateTimeFlow(flow);
            particleSystem.UpdateFlow(flow);
        }
        
        public void UpdateDilation(float dilation)
        {
            currentDilation = dilation;
            quantumBridge.UpdateTimeDilation(dilation);
            particleSystem.UpdateDilation(dilation);
        }
        
        public void UpdateSync(float sync)
        {
            currentSync = sync;
            quantumBridge.UpdateTimeSync(sync);
            particleSystem.UpdateSync(sync);
        }
        
        public void Manipulate()
        {
            float manipulationStrength = currentFlow * FIELD_STRENGTH;
            quantumBridge.ManipulateTime(manipulationStrength);
            particleSystem.Manipulate();
        }
        
        public void Reset()
        {
            float resetCoherence = currentSync * REALITY_COHERENCE;
            quantumBridge.ResetTime(resetCoherence);
            particleSystem.Reset();
        }
    }
}
