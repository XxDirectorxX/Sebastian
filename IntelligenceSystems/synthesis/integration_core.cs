using System;

namespace Sebastian.VoiceCloning.Synthesis
{
    public class IntegrationCore
    {
        private readonly UnifiedProcessor _processor;

        public IntegrationCore()
        {
            _processor = new UnifiedProcessor();
        }

        public double Integrate(double inputState)
        {
            return _processor.ProcessUnified(inputState);
        }
    }
}