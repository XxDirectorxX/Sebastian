using UnityEngine;

public class ContractSealRenderer : MonoBehaviour
{
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float REALITY_COHERENCE = 1.618033988749895f;
    
    [SerializeField] private Material sealMaterial;
    [SerializeField] private ParticleSystem sealParticles;
    private QuantumSystemBridge quantumBridge;
    
    void Start()
    {
        quantumBridge = GetComponent<QuantumSystemBridge>();
        InitializeSealMaterial();
        SetupParticleSystem();
    }

    private void InitializeSealMaterial()
    {
        sealMaterial.SetFloat("_GlowIntensity", FIELD_STRENGTH * 0.1f);
        sealMaterial.SetFloat("_RotationSpeed", REALITY_COHERENCE);
        sealMaterial.SetColor("_GlowColor", new Color(0.545f, 0f, 0f));
    }

    private void SetupParticleSystem()
    {
        var main = sealParticles.main;
        main.startLifetime = REALITY_COHERENCE;
        main.startSpeed = FIELD_STRENGTH * 0.1f;
        
        var emission = sealParticles.emission;
        emission.rateOverTime = FIELD_STRENGTH;
    }

    public void ActivateSeal(float intensity)
    {
        sealMaterial.SetFloat("_GlowIntensity", intensity * FIELD_STRENGTH);
        sealParticles.Play();
        StartSealAnimation();
    }
}
