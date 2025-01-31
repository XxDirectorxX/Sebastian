using System;
using System.Windows;
using System.Windows.Controls;

namespace Sebastian.Panels
{
    public partial class EnergyProjectionPanel : UserControl
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private readonly EnergyProjectionController energyController;
        private readonly QuantumSystemBridge quantumBridge;
        private readonly ParticleSystem particleSystem;
        
        public EnergyProjectionPanel()
        {
            InitializeComponent();
            
            energyController = new EnergyProjectionController();
            quantumBridge = new QuantumSystemBridge();
            particleSystem = new ParticleSystem(EnergyCanvas);
            
            InitializeQuantumComponents();
            SetupEnergyElements();
            ActivateEnergySystems();
        }
        
        private void InitializeQuantumComponents()
        {
            float fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            energyController.ProcessEnergy(fieldEffect);
            quantumBridge.Initialize(fieldEffect);
            particleSystem.Initialize(fieldEffect);
        }
        
        private void SetupEnergyElements()
        {
            IntensitySlider.ValueChanged += OnIntensityChanged;
            FocusSlider.ValueChanged += OnFocusChanged;
            WaveformSlider.ValueChanged += OnWaveformChanged;
            
            ProjectButton.Click += OnProject;
            StabilizeButton.Click += OnStabilize;
        }
        
        private void ActivateEnergySystems()
        {
            energyController.UpdateEnergy(FIELD_STRENGTH);
            quantumBridge.SynchronizeField(REALITY_COHERENCE);
            particleSystem.Start();
        }
        
        private void OnIntensityChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            float intensity = (float)e.NewValue / 100f * FIELD_STRENGTH;
            energyController.UpdateIntensity(intensity);
        }
        
        private void OnFocusChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            float focus = (float)e.NewValue / 100f * FIELD_STRENGTH;
            energyController.UpdateFocus(focus);
        }
        
        private void OnWaveformChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            float waveform = (float)e.NewValue / 100f * REALITY_COHERENCE;
            energyController.UpdateWaveform(waveform);
        }
        
        private void OnProject(object sender, RoutedEventArgs e)
        {
            energyController.Project();
            particleSystem.Activate();
        }
        
        private void OnStabilize(object sender, RoutedEventArgs e)
        {
            quantumBridge.SynchronizeField(REALITY_COHERENCE);
            particleSystem.Stabilize();
        }
    }
}
