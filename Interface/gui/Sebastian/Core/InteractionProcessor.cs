using UnityEngine;
using System.Collections.Generic;

public class InteractionProcessor : MonoBehaviour
{
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float PHI = 1.618033988749895f;
    
    private Dictionary<string, InteractionData> interactionPatterns;
    private QuantumSystemBridge quantumBridge;
    private VoiceController voiceController;
    
    void Start()
    {
        quantumBridge = GetComponent<QuantumSystemBridge>();
        voiceController = GetComponent<VoiceController>();
        InitializeInteractions();
    }

    private void InitializeInteractions()
    {
        interactionPatterns = new Dictionary<string, InteractionData>();
        LoadInteractionPatterns();
        SetupResponseSystem();
    }

    public void ProcessInteraction(string type, float intensity)
    {
        if (interactionPatterns.ContainsKey(type))
        {
            ExecuteInteractionResponse(type, intensity * FIELD_STRENGTH);
            voiceController.SpeakPhrase(interactionPatterns[type].response, intensity);
        }
    }
}
