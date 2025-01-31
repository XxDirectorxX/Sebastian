namespace Sebastian.Core.Base 
{
    /// <summary>
    /// Visualization system interface
    /// </summary>
    public interface IVisualization 
    {
        void Initialize();
        double Process(double inputState);
        double ApplyTransform(double state);
        double StabilizeState(double state);
        double GenerateMetrics(double state);
    }
}
