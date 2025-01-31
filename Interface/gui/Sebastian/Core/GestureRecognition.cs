using UnityEngine;
using System.Collections.Generic;

public class GestureRecognition : MonoBehaviour
{
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float PHI = 1.618033988749895f;
    
    private Dictionary<string, Vector3[]> gesturePatterns;
    private Vector3[] currentGesture;
    private QuantumSystemBridge quantumBridge;
    
    void Start()
    {
        InitializeGestureSystem();
        quantumBridge = GetComponent<QuantumSystemBridge>();
        currentGesture = new Vector3[32];
    }

    private void InitializeGestureSystem()
    {
        gesturePatterns = new Dictionary<string, Vector3[]>();
        LoadGesturePatterns();
        SetupGestureTracking();
    }

    public bool RecognizeGesture(Vector3[] gesturePoints)
    {
        float matchThreshold = FIELD_STRENGTH * 0.1f;
        foreach (var pattern in gesturePatterns)
        {
            if (CompareGesturePattern(gesturePoints, pattern.Value) < matchThreshold)
            {
                ExecuteGestureResponse(pattern.Key);
                return true;
            }
        }
        return false;
    }
}
