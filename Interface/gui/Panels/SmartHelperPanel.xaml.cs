using System.Windows.Controls;
using Sebastian.Core;

namespace Sebastian.Panels
{
    public partial class SmartHelperPanel : UserControl
    {
        private readonly SmartHelperManager helperManager;
        private readonly QuantumSystemBridge quantumBridge;

        public SmartHelperPanel()
        {
            InitializeComponent();
            
            helperManager = new SmartHelperManager(ConversationDisplay);
            quantumBridge = new QuantumSystemBridge();
            
            InitializeQuantumComponents();
            SetupInputHandling();
        }

        private void InitializeQuantumComponents()
        {
            double fieldEffect = Constants.FIELD_STRENGTH * Constants.REALITY_COHERENCE;
            helperManager.Initialize(fieldEffect);
            quantumBridge.Initialize(fieldEffect);
        }

        private void SetupInputHandling()
        {
            InputBox.KeyDown += (s, e) => 
            {
                if (e.Key == System.Windows.Input.Key.Enter)
                {
                    helperManager.ProcessInput(InputBox.Text);
                    InputBox.Clear();
                }
            };
        }
    }
}
