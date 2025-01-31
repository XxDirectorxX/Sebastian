using System.Windows.Controls;

namespace gui.Panels
{
    public partial class SocialFamilyPanel : UserControl
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private SocialFamilyController socialController;
        private QuantumSystemBridge quantumBridge;

        public SocialFamilyPanel()
        {
            InitializeComponent();
            InitializeQuantumComponents();
            SetupSocialElements();
            ActivateSocialSystems();
        }

        private void InitializeQuantumComponents()
        {
            socialController = new SocialFamilyController();
            quantumBridge = new QuantumSystemBridge();
            
            float fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            socialController.ProcessSocial(fieldEffect);
        }

        private void SetupSocialElements()
        {
            InitializeFamilyList();
            InitializeSocialCircles();
            InitializeEventCalendar();
        }

        private void ActivateSocialSystems()
        {
            socialController.UpdateSocial(FIELD_STRENGTH);
            quantumBridge.SynchronizeField(REALITY_COHERENCE);
        }
    }
}
