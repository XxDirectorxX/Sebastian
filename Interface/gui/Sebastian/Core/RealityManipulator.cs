using UnityEngine;

public class RealityManipulator : MonoBehaviour
{
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float REALITY_COHERENCE = 1.618033988749895f;
    
    private QuantumSystemBridge quantumBridge;
    private ParticleSystem[] realityEffects;
    
    void Start()
    {
        quantumBridge = GetComponent<QuantumSystemBridge>();
        InitializeRealitySystem();
        SetupEffects();
    }

    private void InitializeRealitySystem()
    {
        realityEffects = GetComponentsInChildren<ParticleSystem>();
        foreach (var effect in realityEffects)
        {
            ConfigureEffect(effect);
        }
    }

    public void ManipulateReality(Vector3 position, float intensity)
    {
        float fieldStrength = intensity * FIELD_STRENGTH;
        ApplyRealityDistortion(position, fieldStrength);
        EmitQuantumEffects(position);
        SynchronizeWithQuantumField();
    }
}
