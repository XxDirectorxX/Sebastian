using UnityEngine;

public class MainController : MonoBehaviour
{
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float REALITY_COHERENCE = 1.618033988749895f;
    
    private QuantumSystemBridge quantumBridge;
    private MainSystem mainSystem;
    private float[] mainStates;
    
    void Start()
    {
        quantumBridge = GetComponent<QuantumSystemBridge>();
        mainSystem = GetComponent<MainSystem>();
        InitializeMainSystem();
    }

    private void InitializeMainSystem()
    {
        mainStates = new float[64];
        SetupMainPatterns();
        StartMainMonitoring();
    }

    public void ProcessMain(float intensity)
    {
        float fieldEffect = intensity * FIELD_STRENGTH;
        ModifyMainState(fieldEffect);
        HandleMainSequence(intensity);
        SynchronizeMainField();
    }
}
