using UnityEngine;

public class VisualStyleController : MonoBehaviour
{
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float REALITY_COHERENCE = 1.618033988749895f;
    
    private QuantumSystemBridge quantumBridge;
    private VisualStyleSystem styleSystem;
    private float[] styleStates;
    
    void Start()
    {
        quantumBridge = GetComponent<QuantumSystemBridge>();
        styleSystem = GetComponent<VisualStyleSystem>();
        InitializeStyleSystem();
    }

    private void InitializeStyleSystem()
    {
        styleStates = new float[64];
        SetupStylePatterns();
        StartStyleMonitoring();
    }

    public void ProcessStyle(float intensity)
    {
        float fieldEffect = intensity * FIELD_STRENGTH;
        ModifyStyleState(fieldEffect);
        HandleStyleSequence(intensity);
        SynchronizeStyleField();
    }
}
