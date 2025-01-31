using System;

namespace Sebastian.Controllers.System
{
    public class AdvancedController
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private readonly QuantumSystemBridge quantumBridge;
        private readonly ParticleSystem particleSystem;
        
        public AdvancedController()
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
        
        public void Initialize(float fieldStrength)
        {
            quantumBridge.UpdateQuantumState(fieldStrength);
        }
        
        public void StartSystems()
        {
            float startStrength = FIELD_STRENGTH * 0.8f;
            quantumBridge.UpdateFieldStrength(startStrength);
            particleSystem.Start();
        }
        
        public void ManipulateReality()
        {
            float manipulationStrength = FIELD_STRENGTH;
            quantumBridge.ManipulateReality(manipulationStrength);
            particleSystem.Manipulate();
        }
        
        public void ControlTime()
        {
            float timeStrength = FIELD_STRENGTH * 0.9f;
            quantumBridge.ControlTime(timeStrength);
            particleSystem.TimeWarp();
        }
        
        public void ProjectEnergy()
        {
            float energyStrength = FIELD_STRENGTH * 0.8f;
            quantumBridge.ProjectEnergy(energyStrength);
            particleSystem.Project();
        }
        
        public void OptimizeSystems()
        {
            float optimizationStrength = FIELD_STRENGTH;
            quantumBridge.OptimizeField(optimizationStrength);
            particleSystem.Optimize();
        }
    }
}
