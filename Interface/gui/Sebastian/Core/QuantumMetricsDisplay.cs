using UnityEngine;
using UnityEngine.UI;

public class QuantumMetricsDisplay : MonoBehaviour
{
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float REALITY_COHERENCE = 1.618033988749895f;
    
    [SerializeField] private Text fieldStrengthText;
    [SerializeField] private Text coherenceText;
    [SerializeField] private Text stabilityText;
    private QuantumSystemBridge quantumBridge;
    
    void Start()
    {
        quantumBridge = GetComponent<QuantumSystemBridge>();
        InitializeDisplay();
        StartMetricsUpdate();
    }

    private void InitializeDisplay()
    {
        fieldStrengthText.font = Resources.Load<Font>("Fonts/QuantumDisplay");
        coherenceText.font = Resources.Load<Font>("Fonts/QuantumDisplay");
        stabilityText.font = Resources.Load<Font>("Fonts/QuantumDisplay");
        
        UpdateDisplayColors();
    }

    private void UpdateMetrics()
    {
        float currentField = quantumBridge.GetFieldStrength();
        float currentCoherence = quantumBridge.GetCoherenceLevel();
        
        fieldStrengthText.text = $"Field: {currentField:F2}";
        coherenceText.text = $"Coherence: {currentCoherence:F2}";
        stabilityText.text = $"Stability: {CalculateStability():F2}";
    }
}
