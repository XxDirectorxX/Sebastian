namespace Sebastian.Core.Base 
{
    /// <summary>
    /// Reality manipulation interface
    /// </summary>
    public interface IReality 
    {
        void Initialize();
        double Process(double inputState);
        double ApplyTransform(double state);
        double StabilizeState(double state);
        double GenerateMetrics(double state);
    }
}
