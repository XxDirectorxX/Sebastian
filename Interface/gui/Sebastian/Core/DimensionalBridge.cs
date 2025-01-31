using UnityEngine;

public class DimensionalBridge : MonoBehaviour
{
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float REALITY_COHERENCE = 1.618033988749895f;
    
    private QuantumSystemBridge quantumBridge;
    private ContractSealRenderer sealRenderer;
    private ParticleSystem[] bridgeEffects;
    
    void Start()
    {
        quantumBridge = GetComponent<QuantumSystemBridge>();
        sealRenderer = GetComponent<ContractSealRenderer>();
        bridgeEffects = GetComponentsInChildren<ParticleSystem>();
        InitializeBridge();
    }

    private void InitializeBridge()
    {
        SetupBridgeEffects();
        SynchronizeWithQuantumField();
        PrepareTransitionSystem();
    }

    public void CreateBridge(Vector3 location, float intensity)
    {
        float fieldEffect = intensity * FIELD_STRENGTH;
        GenerateBridgePortal(location, fieldEffect);
        EmitBridgeEffects(intensity);
        StabilizeDimensions();
    }
}
