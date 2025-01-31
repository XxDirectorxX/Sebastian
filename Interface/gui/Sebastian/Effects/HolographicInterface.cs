using UnityEngine;
using UnityEngine.UI;

public class HolographicInterface : MonoBehaviour
{
    [SerializeField] private Canvas interfaceCanvas;
    [SerializeField] private Material hologramMaterial;
    [SerializeField] private Image[] holographicElements;
    
    private readonly float FIELD_STRENGTH = 46.97871376f;
    private readonly float REALITY_COHERENCE = 1.618033988749895f;
    
    void Start()
    {
        InitializeHologramMaterial();
        SetupHolographicElements();
    }
    
    private void InitializeHologramMaterial()
    {
        hologramMaterial.SetFloat("_Intensity", FIELD_STRENGTH * 0.1f);
        hologramMaterial.SetFloat("_EdgeGlow", REALITY_COHERENCE);
        hologramMaterial.SetColor("_HologramColor", new Color(0.545f, 0f, 0f, 0.8f));
    }
    
    private void SetupHolographicElements()
    {
        foreach (var element in holographicElements)
        {
            element.material = new Material(hologramMaterial);
            SetupHolographicAnimation(element);
        }
    }
    
    private void SetupHolographicAnimation(Image element)
    {
        var sequence = LeanTween.sequence();
        
        sequence.append(LeanTween.alpha(element.rectTransform, 0.4f, 1f));
        sequence.append(LeanTween.alpha(element.rectTransform, 0.8f, 1f));
        sequence.setLoopPingPong();
    }
    
    public void UpdateHolographicEffect(float intensity)
    {
        foreach (var element in holographicElements)
        {
            var mat = element.material;
            mat.SetFloat("_Intensity", intensity * FIELD_STRENGTH);
            mat.SetFloat("_EdgeGlow", intensity * REALITY_COHERENCE);
        }
    }
}
