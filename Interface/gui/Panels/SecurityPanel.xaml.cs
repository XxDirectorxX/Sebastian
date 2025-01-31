using System.Windows.Controls;
using Sebastian.Core;

namespace Sebastian.Panels
{
    public partial class SecurityPanel : UserControl
    {
        private readonly SecurityManager securityManager;
        private readonly QuantumSystemBridge quantumBridge;

        public SecurityPanel()
        {
            InitializeComponent();
            
            securityManager = new SecurityManager(SecurityFeed);
            quantumBridge = new QuantumSystemBridge();
            
            InitializeQuantumComponents();
            StartSecuritySystems();
        }

        private void InitializeQuantumComponents()
        {
            double fieldEffect = Constants.FIELD_STRENGTH * Constants.REALITY_COHERENCE;
            securityManager.Initialize(fieldEffect);
            quantumBridge.Initialize(fieldEffect);
        }

        private void StartSecuritySystems()
        {
            securityManager.StartMonitoring();
            quantumBridge.SynchronizeField(Constants.REALITY_COHERENCE);
        }
    }
}
