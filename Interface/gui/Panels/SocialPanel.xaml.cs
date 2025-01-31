using System.Windows.Controls;
using Sebastian.Core;

namespace Sebastian.Panels
{
    public partial class SocialPanel : UserControl
    {
        private readonly SocialManager socialManager;
        private readonly QuantumSystemBridge quantumBridge;

        public SocialPanel()
        {
            InitializeComponent();
            
            socialManager = new SocialManager(ContactsList, InteractionSpace);
            quantumBridge = new QuantumSystemBridge();
            
            InitializeQuantumComponents();
            LoadSocialData();
        }

        private void InitializeQuantumComponents()
        {
            double fieldEffect = Constants.FIELD_STRENGTH * Constants.REALITY_COHERENCE;
            socialManager.Initialize(fieldEffect);
            quantumBridge.Initialize(fieldEffect);
        }

        private void LoadSocialData()
        {
            socialManager.LoadContacts();
            ContactsList.SelectionChanged += (s, e) => 
            {
                socialManager.ShowInteraction(ContactsList.SelectedItem);
            };
        }
    }
}
