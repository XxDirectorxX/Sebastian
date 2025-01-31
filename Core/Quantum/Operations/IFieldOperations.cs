namespace Sebastian.Core.Base 
{
    /// <summary>
    /// Field manipulation methods
    /// </summary>
    public interface IFieldOperations 
    {
        void Initialize();
        double Process(double inputState);
        double ApplyTransform(double state);
        double StabilizeState(double state);
        double GenerateMetrics(double state);
    }
}
