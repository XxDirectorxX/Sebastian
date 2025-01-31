using System.Windows.Controls;
using Sebastian.Core;

namespace Sebastian.Panels
{
    public partial class AdvancedPanel : UserControl
    {
        private readonly AdvancedManager advancedManager;
        private readonly QuantumSystemBridge quantumBridge;
        private readonly QuantumFieldVisualizer fieldVisualizer;

        public AdvancedPanel()
        {
            InitializeComponent();
            
            advancedManager = new AdvancedManager();
            quantumBridge = new QuantumSystemBridge();
            fieldVisualizer = new QuantumFieldVisualizer(QuantumVisualizer);
            
            InitializeQuantumComponents();
            SetupAdvancedControls();
        }

        private void InitializeQuantumComponents()
        {
            double fieldEffect = Constants.FIELD_STRENGTH * Constants.REALITY_COHERENCE;
            advancedManager.Initialize(fieldEffect);
            fieldVisualizer.Initialize(fieldEffect);
        }

        private void SetupAdvancedControls()
        {
            advancedManager.StartSystems();
            fieldVisualizer.StartVisualization();
        }
    }
}
