using System.Windows.Controls;
using Sebastian.Core;

namespace Sebastian.Panels
{
    public partial class VoicePanel : UserControl
    {
        private readonly VoiceManager voiceManager;
        private readonly QuantumSystemBridge quantumBridge;
        private readonly VoicePatternVisualizer patternVisualizer;

        public VoicePanel()
        {
            InitializeComponent();
            
            voiceManager = new VoiceManager();
            quantumBridge = new QuantumSystemBridge();
            patternVisualizer = new VoicePatternVisualizer(VoicePatternVisualizer);
            
            InitializeQuantumComponents();
            SetupVoiceControls();
        }

        private void InitializeQuantumComponents()
        {
            double fieldEffect = Constants.FIELD_STRENGTH * Constants.REALITY_COHERENCE;
            voiceManager.Initialize(fieldEffect);
            patternVisualizer.Initialize(fieldEffect);
        }

        private void SetupVoiceControls()
        {
            StartVoiceButton.Click += (s, e) => StartVoiceProcessing();
            StopVoiceButton.Click += (s, e) => StopVoiceProcessing();
        }

        private void StartVoiceProcessing()
        {
            voiceManager.StartVoiceProcessing();
            patternVisualizer.StartVisualization();
        }

        private void StopVoiceProcessing()
        {
            voiceManager.StopVoiceProcessing();
            patternVisualizer.StopVisualization();
        }
    }
}
