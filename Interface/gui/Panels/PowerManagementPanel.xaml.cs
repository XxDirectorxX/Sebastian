using System.Windows.Controls;

namespace gui.Panels
{
    public partial class PowerManagementPanel : UserControl
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private PowerManagementController powerController;
        private QuantumSystemBridge quantumBridge;

        public PowerManagementPanel()
        {
            InitializeComponent();
            InitializeQuantumComponents();
            SetupPowerElements();
            ActivatePowerSystems();
        }

        private void InitializeQuantumComponents()
        {
            powerController = new PowerManagementController();
            quantumBridge = new QuantumSystemBridge();
            
            float fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            powerController.ProcessPower(fieldEffect);
        }

        private void SetupPowerElements()
        {
            InitializeEnergyFlow();
            InitializePowerDistribution();
            InitializeFieldStrength();
            InitializeQuantumResonance();
        }

        private void ActivatePowerSystems()
        {
            powerController.UpdatePower(FIELD_STRENGTH);
            quantumBridge.SynchronizeField(REALITY_COHERENCE);
        }
    }
}
