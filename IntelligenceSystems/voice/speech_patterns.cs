using System;
using System.Numerics;
using Sebastian.Core;
using Sebastian.Core.Quantum;
using Sebastian.Core.Processing;

namespace Sebastian.IntelligenceSystems.Voice
{
    public class SpeechPatterns
    {
        private readonly QuantumProcessor _processor;
        private readonly VoiceProcessor _voiceProcessor;
        
        public SpeechPatterns()
        {
            _processor = new QuantumProcessor();
            _voiceProcessor = new VoiceProcessor();
        }

        public double ProcessSpeech(double inputState)
        {
            var quantumState = _processor.ProcessQuantumState(inputState);
            return _voiceProcessor.ProcessVoice(quantumState);
        }
    }
}