using System.Windows.Controls;
using Sebastian.Core;

namespace Sebastian.Panels
{
    public partial class ControlCenterPanel : UserControl
    {
        private readonly ControlCenterManager controlManager;
        private readonly QuantumSystemBridge quantumBridge;

        public ControlCenterPanel()
        {
            InitializeComponent();
            
            controlManager = new ControlCenterManager(ControlContent);
            quantumBridge = new QuantumSystemBridge();
            
            InitializeQuantumComponents();
            SetupControlButtons();
        }

        private void InitializeQuantumComponents()
        {
            double fieldEffect = Constants.FIELD_STRENGTH * Constants.REALITY_COHERENCE;
            controlManager.Initialize(fieldEffect);
            quantumBridge.Initialize(fieldEffect);
        }

        private void SetupControlButtons()
        {
            HomeButton.Click += (s, e) => controlManager.ShowHomeControls();
            LightsButton.Click += (s, e) => controlManager.ShowLightingControls();
            TempButton.Click += (s, e) => controlManager.ShowTemperatureControls();
            SecurityButton.Click += (s, e) => controlManager.ShowSecurityControls();
        }
    }
}
