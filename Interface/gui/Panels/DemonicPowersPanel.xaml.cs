using System.Windows.Controls;

namespace gui.Panels
{
    public partial class DemonicPowersPanel : UserControl
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private DemonicController demonicController;
        private QuantumSystemBridge quantumBridge;

        public DemonicPowersPanel()
        {
            InitializeComponent();
            InitializeQuantumComponents();
            SetupDemonicElements();
            ActivateDemonicSystems();
        }

        private void InitializeQuantumComponents()
        {
            demonicController = new DemonicController();
            quantumBridge = new QuantumSystemBridge();
            
            float fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            demonicController.ProcessDemonic(fieldEffect);
        }

        private void SetupDemonicElements()
        {
            InitializeContractSeal();
            InitializeRealityWarping();
            InitializePowerManifestation();
            InitializeDemonicEssence();
        }

        private void ActivateDemonicSystems()
        {
            demonicController.UpdateDemonic(FIELD_STRENGTH);
            quantumBridge.SynchronizeField(REALITY_COHERENCE);
        }
    }
}
