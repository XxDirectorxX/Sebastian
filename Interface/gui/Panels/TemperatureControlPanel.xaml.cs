using System;
using System.Windows;
using System.Windows.Controls;

namespace Sebastian.Panels
{
    public partial class TemperatureControlPanel : UserControl
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private readonly TemperatureController temperatureController;
        private readonly QuantumSystemBridge quantumBridge;
        
        public TemperatureControlPanel()
        {
            InitializeComponent();
            
            temperatureController = new TemperatureController();
            quantumBridge = new QuantumSystemBridge();
            
            InitializeQuantumComponents();
            SetupTemperatureElements();
        }
        
        private void InitializeQuantumComponents()
        {
            float fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            temperatureController.Initialize(fieldEffect);
            quantumBridge.Initialize(fieldEffect);
        }
        
        private void SetupTemperatureElements()
        {
            TempSlider.ValueChanged += OnTemperatureChanged;
            HeatMode.Checked += OnModeChanged;
            CoolMode.Checked += OnModeChanged;
            AutoMode.Checked += OnModeChanged;
        }
        
        private void OnTemperatureChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            double temperature = e.NewValue;
            TemperatureDisplay.Text = $"{temperature:F1}°F";
            
            float normalizedTemp = (float)(temperature - 60) / 20f;
            temperatureController.SetTemperature(normalizedTemp);
            UpdateQuantumField(normalizedTemp);
        }
        
        private void OnModeChanged(object sender, RoutedEventArgs e)
        {
            string mode = "Auto";
            if (HeatMode.IsChecked == true) mode = "Heat";
            if (CoolMode.IsChecked == true) mode = "Cool";
            
            temperatureController.SetMode(mode);
            UpdateQuantumField(temperatureController.GetCurrentIntensity());
        }
        
        private void UpdateQuantumField(float intensity)
        {
            float fieldStrength = intensity * FIELD_STRENGTH;
            quantumBridge.UpdateFieldStrength(fieldStrength);
        }
    }
}
