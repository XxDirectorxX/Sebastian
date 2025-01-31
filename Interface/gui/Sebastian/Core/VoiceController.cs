using UnityEngine;
using System;

public class VoiceController : MonoBehaviour
{
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float REALITY_COHERENCE = 1.618033988749895f;
    
    private AudioSource voiceSource;
    private float[] voiceSpectrum;
    private QuantumSystemBridge quantumBridge;
    
    void Start()
    {
        voiceSource = GetComponent<AudioSource>();
        voiceSpectrum = new float[256];
        quantumBridge = GetComponent<QuantumSystemBridge>();
        InitializeVoiceSystem();
    }

    private void InitializeVoiceSystem()
    {
        voiceSource.clip = Resources.Load<AudioClip>("SebastianVoice");
        voiceSource.spatialBlend = 1f;
        voiceSource.minDistance = REALITY_COHERENCE;
        voiceSource.maxDistance = FIELD_STRENGTH;
    }

    public void SpeakPhrase(string phrase, float intensity)
    {
        ProcessVoiceSpectrum();
        ApplyQuantumModulation(intensity);
        PlayVoice();
    }

    private void ProcessVoiceSpectrum()
    {
        voiceSource.GetSpectrumData(voiceSpectrum, 0, FFTWindow.BlackmanHarris);
        for (int i = 0; i < voiceSpectrum.Length; i++)
        {
            voiceSpectrum[i] *= FIELD_STRENGTH;
        }
    }
}
