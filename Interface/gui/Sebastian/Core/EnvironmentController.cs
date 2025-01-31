using UnityEngine;
using System.Collections.Generic;

public class EnvironmentController : MonoBehaviour
{
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float REALITY_COHERENCE = 1.618033988749895f;
    
    private Dictionary<string, EnvironmentState> environmentStates;
    private QuantumSystemBridge quantumBridge;
    private ParticleSystem[] environmentEffects;
    
    void Start()
    {
        quantumBridge = GetComponent<QuantumSystemBridge>();
        environmentEffects = GetComponentsInChildren<ParticleSystem>();
        InitializeEnvironment();
    }

    private void InitializeEnvironment()
    {
        environmentStates = new Dictionary<string, EnvironmentState>();
        LoadEnvironmentStates();
        SetupEnvironmentEffects();
    }

    public void ModifyEnvironment(string state, float intensity)
    {
        float fieldEffect = intensity * FIELD_STRENGTH;
        ApplyEnvironmentState(state, fieldEffect);
        UpdateQuantumField(fieldEffect);
        EmitEnvironmentParticles(intensity);
    }
}