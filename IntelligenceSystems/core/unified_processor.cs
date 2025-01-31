using System;

namespace Sebastian.VoiceCloning.Core
{
    public class UnifiedProcessor
    {
        private readonly VoiceSynthesis _synthesis;

        public UnifiedProcessor()
        {
            _synthesis = new VoiceSynthesis();
        }

        public double ProcessUnified(double inputState)
        {
            return _synthesis.ProcessVoice(inputState);
        }
    }
}
