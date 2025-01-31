using UnityEngine;
using System.Collections.Generic;

public class QuantumFieldVisualizer : MonoBehaviour
{
    [SerializeField] private ParticleSystem quantumParticles;
    [SerializeField] private Material fieldMaterial;
    
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float PHI = 1.618033988749895f;
    private List<ParticleSystem.Particle> particles;
    
    void Start()
    {
        particles = new List<ParticleSystem.Particle>();
        InitializeFieldMaterial();
        SetupParticleSystem();
    }
    
    private void InitializeFieldMaterial()
    {
        fieldMaterial.SetFloat("_FieldStrength", FIELD_STRENGTH);
        fieldMaterial.SetFloat("_Coherence", PHI);
        fieldMaterial.SetColor("_GlowColor", new Color(0.545f, 0f, 0f));
    }
    
    private void SetupParticleSystem()
    {
        var main = quantumParticles.main;
        main.startSpeed = FIELD_STRENGTH * 0.1f;
        main.startSize = PHI * 0.1f;
        
        var emission = quantumParticles.emission;
        emission.rateOverTime = FIELD_STRENGTH;
        
        var shape = quantumParticles.shape;
        shape.radius = PHI;
    }
    
    public void UpdateFieldVisualization(float fieldIntensity)
    {
        int particleCount = quantumParticles.GetParticles(particles);
        
        for (int i = 0; i < particleCount; i++)
        {
            var particle = particles[i];
            particle.velocity *= fieldIntensity * FIELD_STRENGTH;
            particle.startColor = Color.Lerp(
                Color.red * 0.5f, 
                Color.black, 
                Mathf.PingPong(Time.time * PHI, 1)
            );
            particles[i] = particle;
        }
        
        quantumParticles.SetParticles(particles, particleCount);
    }
}
