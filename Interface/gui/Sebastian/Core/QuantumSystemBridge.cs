using System;
using System.Windows;

namespace Sebastian.Core
{
    /// <summary>
    /// Manages quantum field operations and system synchronization
    /// </summary>
    public class QuantumSystemBridge 
    {
        // Quantum constants
        private const double FIELD_STRENGTH = 46.97871376;
        private const double REALITY_COHERENCE = 1.618033988749895;
        private const double PHI = 1.618033988749895;

        // System components
        private readonly ParticleSystem particleSystem;
        private readonly FieldVisualizer fieldVisualizer;
        private readonly StatusMonitor statusMonitor;

        public QuantumSystemBridge(ParticleSystem particleSystem, FieldVisualizer fieldVisualizer)
        {
            this.particleSystem = particleSystem;
            this.fieldVisualizer = fieldVisualizer;
            this.statusMonitor = new StatusMonitor();
        }

        public void Initialize()
        {
            try
            {
                double fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
                particleSystem.Initialize(fieldEffect);
                fieldVisualizer.Initialize(fieldEffect);
                statusMonitor.Initialize(fieldEffect);
            }
            catch (Exception ex)
            {
                LogError("Quantum initialization failed", ex);
                throw;
            }
        }

        public void UpdateFieldStrength(double intensity)
        {
            double fieldStrength = intensity * FIELD_STRENGTH;
            particleSystem.UpdateIntensity(fieldStrength);
            fieldVisualizer.UpdateField(fieldStrength);
        }

        private void LogError(string message, Exception ex)
        {
            statusMonitor.LogError($"QuantumSystemBridge: {message}", ex);
        }
    }
}
