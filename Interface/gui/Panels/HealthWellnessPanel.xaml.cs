using System.Windows.Controls;

namespace gui.Panels
{
    public partial class HealthWellnessPanel : UserControl
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private HealthWellnessController healthController;
        private QuantumSystemBridge quantumBridge;

        public HealthWellnessPanel()
        {
            InitializeComponent();
            InitializeQuantumComponents();
            SetupHealthElements();
            ActivateHealthSystems();
        }

        private void InitializeQuantumComponents()
        {
            healthController = new HealthWellnessController();
            quantumBridge = new QuantumSystemBridge();
            
            float fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            healthController.ProcessHealth(fieldEffect);
        }

        private void SetupHealthElements()
        {
            InitializeHealthMonitoring();
            InitializeWellnessTracking();
            InitializePTSDSupport();
        }

        private void ActivateHealthSystems()
        {
            healthController.UpdateHealth(FIELD_STRENGTH);
            quantumBridge.SynchronizeField(REALITY_COHERENCE);
        }
    }
}
