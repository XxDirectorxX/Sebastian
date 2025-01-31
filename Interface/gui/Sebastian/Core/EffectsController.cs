using UnityEngine;

public class EffectsController : MonoBehaviour
{
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float REALITY_COHERENCE = 1.618033988749895f;
    
    private QuantumSystemBridge quantumBridge;
    private ParticleSystem[] quantumEffects;
    private float[] effectStates;
    
    void Start()
    {
        quantumBridge = GetComponent<QuantumSystemBridge>();
        quantumEffects = GetComponentsInChildren<ParticleSystem>();
        InitializeEffects();
    }

    private void InitializeEffects()
    {
        effectStates = new float[64];
        SetupEffectSystems();
        StartEffectsMonitoring();
    }

    public void UpdateEffects(float intensity)
    {
        float fieldEffect = intensity * FIELD_STRENGTH;
        ModifyEffectStates(fieldEffect);
        EmitQuantumEffects(intensity);
        SynchronizeEffectField();
    }
}
