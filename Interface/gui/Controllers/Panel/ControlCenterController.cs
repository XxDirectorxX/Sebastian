using System;
using System.Windows.Controls;
using Sebastian.Core;
using Sebastian.Core.Processing;
using Sebastian.Core.Quantum;

namespace Sebastian.Interface.Controllers
{
    public class ControlCenterController
    {
        private readonly Panel _controlPanel;
        private readonly QuantumProcessor _processor;
        private readonly SystemController _systemController;

        public ControlCenterController(Panel controlPanel)
        {
            _controlPanel = controlPanel;
            _processor = new QuantumProcessor();
            _systemController = new SystemController();
            InitializeControls();
        }

        private void InitializeControls()
        {
            _systemController.InitializeQuantumSystem();
            _processor.InitializeQuantumState();
        }

        public void ShowLightingControls(ContentControl content)
        {
            content.Content = new LightingControlPanel();
            UpdateFieldState(FIELD_STRENGTH * 0.8f);
        }
        
        public void ShowTemperatureControls(ContentControl content)
        {
            content.Content = new TemperatureControlPanel();
            UpdateFieldState(FIELD_STRENGTH * 0.9f);
        }
        
        public void ShowSecurityControls(ContentControl content)
        {
            content.Content = new SecurityControlPanel();
            UpdateFieldState(FIELD_STRENGTH);
        }
        
        public void TurnOffAllSystems()
        {
            quantumBridge.DeactivateField();
            particleSystem.Stop();
        }
        
        public void OptimizeSystems(float strength)
        {
            quantumBridge.OptimizeField(strength);
            particleSystem.Optimize();
        }
        
        private void UpdateFieldState(float strength)
        {
            quantumBridge.UpdateFieldStrength(strength);
            particleSystem.UpdateIntensity(strength);
        }
    }
}