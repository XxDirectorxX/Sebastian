using UnityEngine;
using System;
using System.Collections.Generic;

public class SebastianAnimationController : MonoBehaviour 
{
    [SerializeField] private Animator sebastianAnimator;
    [SerializeField] private ParticleSystem contractSealParticles;
    [SerializeField] private SkinnedMeshRenderer sebastianRenderer;
    
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float REALITY_COHERENCE = 1.618033988749895f;
    
    private Dictionary<string, AnimationClip> animationClips;
    private QuantumFieldController quantumController;
    
    void Start()
    {
        quantumController = GetComponent<QuantumFieldController>();
        InitializeAnimations();
        SetupBlendShapes();
        SyncWithQuantumField();
    }
    
    private void InitializeAnimations()
    {
        animationClips = new Dictionary<string, AnimationClip>();
        var clips = sebastianAnimator.runtimeAnimatorController.animationClips;
        foreach (var clip in clips)
        {
            animationClips.Add(clip.name, clip);
        }
    }
    
    public void ExecuteAnimation(string animName, float quantumInfluence)
    {
        if (animationClips.ContainsKey(animName))
        {
            float fieldStrength = quantumController.GetCurrentFieldStrength();
            sebastianAnimator.SetFloat("FieldStrength", fieldStrength * FIELD_STRENGTH);
            sebastianAnimator.SetFloat("Coherence", REALITY_COHERENCE);
            sebastianAnimator.Play(animName);
            
            if (quantumInfluence > 0)
            {
                EmitContractSealParticles(quantumInfluence);
            }
        }
    }
    
    private void EmitContractSealParticles(float intensity)
    {
        var emission = contractSealParticles.emission;
        emission.rateOverTime = intensity * FIELD_STRENGTH;
        contractSealParticles.Play();
    }
}
