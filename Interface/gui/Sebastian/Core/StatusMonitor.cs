using UnityEngine;

public class StatusMonitor : MonoBehaviour
{
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float REALITY_COHERENCE = 1.618033988749895f;
    
    private QuantumSystemBridge quantumBridge;
    private QuantumMetricsDisplay metricsDisplay;
    private float[] statusHistory;
    
    void Start()
    {
        quantumBridge = GetComponent<QuantumSystemBridge>();
        metricsDisplay = GetComponent<QuantumMetricsDisplay>();
        InitializeMonitor();
    }

    private void InitializeMonitor()
    {
        statusHistory = new float[64];
        StartMonitoring();
    }

    private void UpdateStatus()
    {
        float currentField = quantumBridge.GetFieldStrength();
        float currentCoherence = quantumBridge.GetCoherenceLevel();
        
        UpdateStatusHistory(currentField);
        AnalyzeSystemStatus(currentCoherence);
        metricsDisplay.UpdateMetrics();
    }
}
