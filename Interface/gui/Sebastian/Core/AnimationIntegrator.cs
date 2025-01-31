using UnityEngine;
using System.Collections.Generic;

public class AnimationIntegrator : MonoBehaviour
{
    [SerializeField] private Animator sebastianAnimator;
    [SerializeField] private ParticleSystem[] quantumEffects;
    
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float PHI = 1.618033988749895f;
    
    private Dictionary<string, AnimationState> animationStates;
    private QuantumSystemBridge quantumBridge;
    
    void Start()
    {
        quantumBridge = GetComponent<QuantumSystemBridge>();
        InitializeAnimationStates();
        SetupQuantumEffects();
    }
    
    private void InitializeAnimationStates()
    {
        animationStates = new Dictionary<string, AnimationState>();
        foreach (AnimationState state in sebastianAnimator)
        {
            animationStates.Add(state.name, state);
            state.speed = FIELD_STRENGTH * 0.01f;
        }
    }
    
    private void SetupQuantumEffects()
    {
        foreach (var effect in quantumEffects)
        {
            var main = effect.main;
            main.startLifetime = PHI;
            main.startSpeed = FIELD_STRENGTH * 0.1f;
            
            var emission = effect.emission;
            emission.rateOverTime = FIELD_STRENGTH;
        }
    }
    
    public void PlayAnimation(string animName, float quantumInfluence)
    {
        if (animationStates.ContainsKey(animName))
        {
            var state = animationStates[animName];
            state.speed = FIELD_STRENGTH * quantumInfluence;
            sebastianAnimator.Play(animName);
            
            foreach (var effect in quantumEffects)
            {
                effect.Play();
            }
        }
    }
}
