using UnityEngine;
using System.Collections.Generic;

public class SecurityController : MonoBehaviour
{
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float REALITY_COHERENCE = 1.618033988749895f;
    
    private Dictionary<string, SecurityProtocol> securityProtocols;
    private QuantumSystemBridge quantumBridge;
    private ContractSealRenderer sealRenderer;
    
    void Start()
    {
        quantumBridge = GetComponent<QuantumSystemBridge>();
        sealRenderer = GetComponent<ContractSealRenderer>();
        InitializeSecurity();
    }

    private void InitializeSecurity()
    {
        securityProtocols = new Dictionary<string, SecurityProtocol>();
        LoadSecurityProtocols();
        ActivateBaseProtection();
    }

    public void EnforceSecurity(string protocol, float intensity)
    {
        if (securityProtocols.ContainsKey(protocol))
        {
            sealRenderer.ActivateSeal(intensity);
            ExecuteSecurityProtocol(protocol, intensity * FIELD_STRENGTH);
        }
    }
}
