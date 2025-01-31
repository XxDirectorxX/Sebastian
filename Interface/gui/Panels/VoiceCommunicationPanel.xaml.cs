using System.Windows.Controls;

namespace gui.Panels
{
    public partial class VoiceCommunicationPanel : UserControl
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private VoiceCommunicationController voiceController;
        private QuantumSystemBridge quantumBridge;

        public VoiceCommunicationPanel()
        {
            InitializeComponent();
            InitializeQuantumComponents();
            SetupVoiceElements();
            ActivateVoiceSystems();
        }

        private void InitializeQuantumComponents()
        {
            voiceController = new VoiceCommunicationController();
            quantumBridge = new QuantumSystemBridge();
            
            float fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            voiceController.ProcessVoice(fieldEffect);
        }

        private void SetupVoiceElements()
        {
            InitializeVoiceChat();
            InitializeVideoChat();
            InitializeTextChat();
            InitializeVoiceProcessing();
        }

        private void ActivateVoiceSystems()
        {
            voiceController.UpdateVoice(FIELD_STRENGTH);
            quantumBridge.SynchronizeField(REALITY_COHERENCE);
        }
    }
}
