using System;
using System.Windows.Media;

namespace Sebastian.Core
{
    public class VoiceManager
    {
        private readonly QuantumVoiceProcessor voiceProcessor;
        private bool isProcessing;

        public VoiceManager()
        {
            voiceProcessor = new QuantumVoiceProcessor();
            isProcessing = false;
        }

        public void Initialize(double fieldStrength)
        {
            voiceProcessor.Initialize(fieldStrength);
        }

        public void StartVoiceProcessing()
        {
            if (!isProcessing)
            {
                isProcessing = true;
                voiceProcessor.StartProcessing();
            }
        }

        public void StopVoiceProcessing()
        {
            if (isProcessing)
            {
                isProcessing = false;
                voiceProcessor.StopProcessing();
            }
        }
    }
}
