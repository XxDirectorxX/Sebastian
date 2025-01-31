using UnityEngine;

public class TimeManipulator : MonoBehaviour
{
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float REALITY_COHERENCE = 1.618033988749895f;
    
    private QuantumSystemBridge quantumBridge;
    private ParticleSystem[] timeEffects;
    private float currentTimeScale;
    
    void Start()
    {
        quantumBridge = GetComponent<QuantumSystemBridge>();
        timeEffects = GetComponentsInChildren<ParticleSystem>();
        InitializeTimeSystem();
    }

    private void InitializeTimeSystem()
    {
        currentTimeScale = 1.0f;
        SetupTimeEffects();
        SynchronizeWithQuantumField();
    }

    public void ManipulateTime(float scale, float intensity)
    {
        float fieldEffect = intensity * FIELD_STRENGTH;
        ApplyTimeScale(scale, fieldEffect);
        EmitTimeEffects(intensity);
        UpdateQuantumCoherence();
    }
}
