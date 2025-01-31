using System;
using Sebastian.Core;

namespace Sebastian.IntelligenceSystems.Voice
{
    public class VoiceProcessor
    {
        private readonly SpeechPatterns _speechPatterns;

        public VoiceProcessor()
        {
            _speechPatterns = new SpeechPatterns();
        }

        public double ProcessVoice(double inputState)
        {
            return _speechPatterns.ProcessSpeech(inputState);
        }
    }
}
