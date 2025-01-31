using System;
using System.Windows;
using System.Windows.Controls;

namespace Sebastian.Panels
{
    public partial class ContractSealPanel : UserControl
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private readonly ContractSealController sealController;
        private readonly QuantumSystemBridge quantumBridge;
        private readonly ParticleSystem particleSystem;
        
        public ContractSealPanel()
        {
            InitializeComponent();
            
            sealController = new ContractSealController();
            quantumBridge = new QuantumSystemBridge();
            particleSystem = new ParticleSystem(SealCanvas);
            
            InitializeQuantumComponents();
            SetupSealElements();
            ActivateSealSystems();
        }
        
        private void InitializeQuantumComponents()
        {
            float fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            sealController.ProcessSeal(fieldEffect);
            quantumBridge.Initialize(fieldEffect);
            particleSystem.Initialize(fieldEffect);
        }
        
        private void SetupSealElements()
        {
            PowerSlider.ValueChanged += OnPowerChanged;
            IntensitySlider.ValueChanged += OnIntensityChanged;
            ResonanceSlider.ValueChanged += OnResonanceChanged;
            
            ActivateButton.Click += OnActivate;
            SynchronizeButton.Click += OnSynchronize;
        }
        
        private void ActivateSealSystems()
        {
            sealController.UpdateSeal(FIELD_STRENGTH);
            quantumBridge.SynchronizeField(REALITY_COHERENCE);
            particleSystem.Start();
        }
        
        private void OnPowerChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            float power = (float)e.NewValue / 100f * FIELD_STRENGTH;
            sealController.UpdatePower(power);
        }
        
        private void OnIntensityChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            float intensity = (float)e.NewValue / 100f * FIELD_STRENGTH;
            sealController.UpdateIntensity(intensity);
        }
        
        private void OnResonanceChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            float resonance = (float)e.NewValue / 100f * REALITY_COHERENCE;
            sealController.UpdateResonance(resonance);
        }
        
        private void OnActivate(object sender, RoutedEventArgs e)
        {
            sealController.Activate();
            particleSystem.Activate();
        }
        
        private void OnSynchronize(object sender, RoutedEventArgs e)
        {
            quantumBridge.SynchronizeField(REALITY_COHERENCE);
            particleSystem.Synchronize();
        }
    }
}
