using System.Windows.Controls;

namespace gui.Panels
{
    public partial class SettingsPanel : UserControl
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private SettingsController settingsController;
        private QuantumSystemBridge quantumBridge;

        public SettingsPanel()
        {
            InitializeComponent();
            InitializeQuantumComponents();
            SetupSettingsElements();
            ActivateSettingsSystems();
        }

        private void InitializeQuantumComponents()
        {
            settingsController = new SettingsController();
            quantumBridge = new QuantumSystemBridge();
            
            float fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            settingsController.ProcessSettings(fieldEffect);
        }

        private void SetupSettingsElements()
        {
            InitializeThemeSettings();
            InitializeFunctionalitySettings();
            InitializePersonalPreferences();
        }

        private void ActivateSettingsSystems()
        {
            settingsController.UpdateSettings(FIELD_STRENGTH);
            quantumBridge.SynchronizeField(REALITY_COHERENCE);
        }
    }
}
