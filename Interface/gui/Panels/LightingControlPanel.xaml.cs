using System;
using System.Windows.Controls;

namespace Sebastian.Panels
{
    public partial class LightingControlPanel : UserControl
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private readonly LightingController lightingController;
        private readonly QuantumSystemBridge quantumBridge;
        
        public LightingControlPanel()
        {
            InitializeComponent();
            
            lightingController = new LightingController();
            quantumBridge = new QuantumSystemBridge();
            
            InitializeQuantumComponents();
            SetupLightingElements();
        }
        
        private void InitializeQuantumComponents()
        {
            float fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            lightingController.Initialize(fieldEffect);
            quantumBridge.Initialize(fieldEffect);
        }
        
        private void SetupLightingElements()
        {
            RoomSelector.SelectionChanged += OnRoomChanged;
            BrightnessSlider.ValueChanged += OnBrightnessChanged;
            ColorTempSlider.ValueChanged += OnColorTempChanged;
        }
        
        private void OnRoomChanged(object sender, SelectionChangedEventArgs e)
        {
            string selectedRoom = (RoomSelector.SelectedItem as ComboBoxItem)?.Content.ToString();
            if (selectedRoom != null)
            {
                lightingController.SelectRoom(selectedRoom);
                UpdateQuantumField();
            }
        }
        
        private void OnBrightnessChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            float brightness = (float)e.NewValue / 100f;
            lightingController.SetBrightness(brightness);
            UpdateQuantumField();
        }
        
        private void OnColorTempChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            float colorTemp = (float)e.NewValue / 100f;
            lightingController.SetColorTemperature(colorTemp);
            UpdateQuantumField();
        }
        
        private void UpdateQuantumField()
        {
            float fieldStrength = lightingController.GetCurrentIntensity() * FIELD_STRENGTH;
            quantumBridge.UpdateFieldStrength(fieldStrength);
        }
    }
}
