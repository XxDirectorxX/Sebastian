using UnityEngine;
using System.Collections.Generic;

public class CommandExecutor : MonoBehaviour
{
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float REALITY_COHERENCE = 1.618033988749895f;
    
    private Dictionary<string, System.Action<float>> commandActions;
    private QuantumSystemBridge quantumBridge;
    private ContractSealRenderer sealRenderer;
    
    void Start()
    {
        quantumBridge = GetComponent<QuantumSystemBridge>();
        sealRenderer = GetComponent<ContractSealRenderer>();
        InitializeCommands();
    }

    private void InitializeCommands()
    {
        commandActions = new Dictionary<string, System.Action<float>>();
        SetupCommandResponses();
        RegisterDefaultCommands();
    }

    public void ExecuteCommand(string command, float intensity)
    {
        if (commandActions.ContainsKey(command))
        {
            sealRenderer.ActivateSeal(intensity);
            commandActions[command].Invoke(intensity * FIELD_STRENGTH);
        }
    }
}
