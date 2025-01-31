using UnityEngine;

public class VisualizationController : MonoBehaviour
{
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float REALITY_COHERENCE = 1.618033988749895f;
    
    private QuantumSystemBridge quantumBridge;
    private ParticleSystem[] visualEffects;
    private float[] visualStates;
    
    void Start()
    {
        quantumBridge = GetComponent<QuantumSystemBridge>();
        visualEffects = GetComponentsInChildren<ParticleSystem>();
        InitializeVisualization();
    }

    private void InitializeVisualization()
    {
        visualStates = new float[64];
        SetupVisualEffects();
        StartVisualizationSystem();
    }

    public void UpdateVisualization(float intensity)
    {
        float fieldEffect = intensity * FIELD_STRENGTH;
        ModifyVisualState(fieldEffect);
        EmitVisualEffects(intensity);
        SynchronizeVisualField();
    }
}
