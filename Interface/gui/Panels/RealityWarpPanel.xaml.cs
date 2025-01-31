using System;
using System.Windows;
using System.Windows.Controls;

namespace Sebastian.Panels
{
    public partial class RealityWarpPanel : UserControl
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private readonly RealityWarpController warpController;
        private readonly QuantumSystemBridge quantumBridge;
        private readonly ParticleSystem particleSystem;
        
        public RealityWarpPanel()
        {
            InitializeComponent();
            
            warpController = new RealityWarpController();
            quantumBridge = new QuantumSystemBridge();
            particleSystem = new ParticleSystem(RealityCanvas);
            
            InitializeQuantumComponents();
            SetupWarpElements();
            ActivateWarpSystems();
        }
        
        private void InitializeQuantumComponents()
        {
            float fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            warpController.ProcessWarp(fieldEffect);
            quantumBridge.Initialize(fieldEffect);
            particleSystem.Initialize(fieldEffect);
        }
        
        private void SetupWarpElements()
        {
            DistortionSlider.ValueChanged += OnDistortionChanged;
            StabilitySlider.ValueChanged += OnStabilityChanged;
            CoherenceSlider.ValueChanged += OnCoherenceChanged;
            
            WarpButton.Click += OnWarp;
            NormalizeButton.Click += OnNormalize;
        }
        
        private void ActivateWarpSystems()
        {
            warpController.UpdateWarp(FIELD_STRENGTH);
            quantumBridge.SynchronizeField(REALITY_COHERENCE);
            particleSystem.Start();
        }
        
        private void OnDistortionChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            float distortion = (float)e.NewValue / 100f * FIELD_STRENGTH;
            warpController.UpdateDistortion(distortion);
        }
        
        private void OnStabilityChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            float stability = (float)e.NewValue / 100f * FIELD_STRENGTH;
            warpController.UpdateStability(stability);
        }
        
        private void OnCoherenceChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            float coherence = (float)e.NewValue / 100f * REALITY_COHERENCE;
            warpController.UpdateCoherence(coherence);
        }
        
        private void OnWarp(object sender, RoutedEventArgs e)
        {
            warpController.Warp();
            particleSystem.Activate();
        }
        
        private void OnNormalize(object sender, RoutedEventArgs e)
        {
            quantumBridge.SynchronizeField(REALITY_COHERENCE);
            particleSystem.Normalize();
        }
    }
}
