using System.Windows.Controls;
using Sebastian.Core;

namespace Sebastian.Panels
{
    public partial class HealthPanel : UserControl
    {
        private readonly HealthManager healthManager;
        private readonly QuantumSystemBridge quantumBridge;
        private readonly QuantumFieldVisualizer fieldVisualizer;

        public HealthPanel()
        {
            InitializeComponent();
            
            healthManager = new HealthManager();
            quantumBridge = new QuantumSystemBridge();
            fieldVisualizer = new QuantumFieldVisualizer(HealthVisualizer);
            
            InitializeQuantumComponents();
            SetupHealthMonitoring();
        }

        private void InitializeQuantumComponents()
        {
            double fieldEffect = Constants.FIELD_STRENGTH * Constants.REALITY_COHERENCE;
            healthManager.Initialize(fieldEffect);
            fieldVisualizer.Initialize(fieldEffect);
        }

        private void SetupHealthMonitoring()
        {
            healthManager.StartMonitoring();
            fieldVisualizer.StartVisualization();
        }
    }
}
