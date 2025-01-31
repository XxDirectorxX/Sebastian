using UnityEngine;

public class EnergyProjector : MonoBehaviour
{
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float REALITY_COHERENCE = 1.618033988749895f;
    
    private QuantumSystemBridge quantumBridge;
    private ParticleSystem[] energyEffects;
    private LineRenderer energyBeam;
    
    void Start()
    {
        quantumBridge = GetComponent<QuantumSystemBridge>();
        energyEffects = GetComponentsInChildren<ParticleSystem>();
        energyBeam = GetComponent<LineRenderer>();
        InitializeProjector();
    }

    private void InitializeProjector()
    {
        SetupEnergyEffects();
        ConfigureEnergyBeam();
        SynchronizeWithQuantumField();
    }

    public void ProjectEnergy(Vector3 target, float intensity)
    {
        float fieldEffect = intensity * FIELD_STRENGTH;
        GenerateEnergyBeam(target, fieldEffect);
        EmitEnergyEffects(intensity);
        StabilizeProjection();
    }
}
