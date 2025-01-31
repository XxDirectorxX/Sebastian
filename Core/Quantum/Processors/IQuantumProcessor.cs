namespace Sebastian.Core.Base 
{
    /// <summary>
    /// Quantum processing operations
    /// </summary>
    public interface IQuantumProcessor 
    {
        void Initialize();
        double Process(double inputState);
        double ApplyTransform(double state);
        double StabilizeState(double state);
        double GenerateMetrics(double state);
    }
}
