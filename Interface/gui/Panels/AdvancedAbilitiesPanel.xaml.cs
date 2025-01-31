using System.Windows.Controls;

namespace gui.Panels
{
    public partial class AdvancedAbilitiesPanel : UserControl
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private AdvancedAbilitiesController abilitiesController;
        private QuantumSystemBridge quantumBridge;

        public AdvancedAbilitiesPanel()
        {
            InitializeComponent();
            InitializeQuantumComponents();
            SetupAbilityElements();
            ActivateAbilitySystems();
        }

        private void InitializeQuantumComponents()
        {
            abilitiesController = new AdvancedAbilitiesController();
            quantumBridge = new QuantumSystemBridge();
            
            float fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            abilitiesController.ProcessAbilities(fieldEffect);
        }

        private void SetupAbilityElements()
        {
            InitializeRealityManipulation();
            InitializeTimeControl();
            InitializeEnergyProjection();
            InitializeSystemOptimization();
        }

        private void ActivateAbilitySystems()
        {
            abilitiesController.UpdateAbilities(FIELD_STRENGTH);
            quantumBridge.SynchronizeField(REALITY_COHERENCE);
        }
    }
}
