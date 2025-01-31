using UnityEngine;
using System.Collections.Generic;

public class SystemInitializer : MonoBehaviour
{
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float REALITY_COHERENCE = 1.618033988749895f;
    
    private QuantumSystemIntegrator systemIntegrator;
    private ControllerOrchestrator orchestrator;
    private Dictionary<string, bool> initializationStates;
    
    void Awake()
    {
        systemIntegrator = GetComponent<QuantumSystemIntegrator>();
        orchestrator = GetComponent<ControllerOrchestrator>();
        InitializeSystem();
    }

    private void InitializeSystem()
    {
        initializationStates = new Dictionary<string, bool>();
        InitializeQuantumCore();
        InitializeControllers();
        EstablishConnections();
        LaunchSebastianInterface();
    }

    private void InitializeQuantumCore()
    {
        float fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
        InitializeQuantumFields(fieldEffect);
        SetupQuantumBridge();
        SynchronizeQuantumStates();
    }

    private void InitializeControllers()
    {
        // Initialize all controllers in correct sequence
        InitializeController<DemonicController>("Demonic");
        InitializeController<PowerManifestationController>("Power");
        InitializeController<RealityWarpController>("Reality");
        // ... Initialize all 35 controllers
    }

    private void LaunchSebastianInterface()
    {
        ActivateQuantumFields();
        SynchronizeAllSystems();
        InitializeGUI();
        BeginSystemOperations();
    }

    private void BeginSystemOperations()
    {
        systemIntegrator.BeginOperation();
        orchestrator.StartOrchestration();
        EmitSystemReadySignal();
    }
}
