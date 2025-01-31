using UnityEngine;

public class QuantumFieldGenerator : MonoBehaviour
{
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float PHI = 1.618033988749895f;
    
    [SerializeField] private ParticleSystem fieldParticles;
    [SerializeField] private Material fieldMaterial;
    private QuantumSystemBridge quantumBridge;
    
    void Start()
    {
        quantumBridge = GetComponent<QuantumSystemBridge>();
        InitializeFieldSystem();
        StartFieldGeneration();
    }

    private void InitializeFieldSystem()
    {
        SetupFieldMaterial();
        ConfigureParticleSystem();
    }

    private void SetupFieldMaterial()
    {
        fieldMaterial.SetFloat("_FieldStrength", FIELD_STRENGTH);
        fieldMaterial.SetFloat("_WaveLength", PHI);
        fieldMaterial.SetColor("_FieldColor", new Color(0.545f, 0f, 0f, 0.5f));
    }

    public void GenerateField(float intensity)
    {
        UpdateFieldParameters(intensity);
        EmitFieldParticles();
        SynchronizeWithQuantumBridge();
    }
}
