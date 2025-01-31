using System;

namespace Sebastian.Controllers.Voice
{
    public class VoicePatternController
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private readonly QuantumSystemBridge quantumBridge;
        private readonly ParticleSystem particleSystem;
        
        public VoicePatternController()
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
        
        public void ProcessVoicePattern(float intensity)
        {
            quantumBridge.SynchronizeField(intensity);
            particleSystem.UpdateIntensity(intensity);
        }
        
        public void UpdatePattern(float strength)
        {
            quantumBridge.UpdateQuantumState(strength);
            particleSystem.UpdateParticles(strength);
        }
        
        public void AnalyzePattern(float[] voiceData)
        {
            float analysisStrength = FIELD_STRENGTH;
            quantumBridge.AnalyzeField(analysisStrength);
            particleSystem.Analyze(voiceData);
        }
        
        public void SynchronizePattern()
        {
            float syncCoherence = REALITY_COHERENCE;
            quantumBridge.SynchronizeField(syncCoherence);
            particleSystem.Synchronize();
        }
    }
}
