using System;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Effects;

namespace Sebastian
{
    public class QuantumEffectRenderer
    {
        private readonly MainWindow window;
        private readonly double fieldStrength = 46.97871376;
        private readonly double phi = 1.618033988749895;

        public QuantumEffectRenderer(MainWindow mainWindow)
        {
            window = mainWindow;
            InitializeEffects();
        }

        private void InitializeEffects()
        {
            CreateContractSealGlow();
            CreateQuantumFieldDistortion();
            CreateHolographicShimmer();
        }

        private void CreateContractSealGlow()
        {
            var glowEffect = new BlurEffect
            {
                Radius = 20,
                KernelType = KernelType.Gaussian
            };
            
            window.ContractSealLayer.Effect = glowEffect;
        }

        private void CreateQuantumFieldDistortion()
        {
            var distortionEffect = new BlurEffect
            {
                Radius = 5,
                RenderingBias = RenderingBias.Performance
            };

            window.EnvironmentLayer.Effect = distortionEffect;
        }

        private void CreateHolographicShimmer()
        {
            var shimmerEffect = new DropShadowEffect
            {
                Color = Color.FromRgb(191, 155, 48), // #BF9B30
                Direction = 0,
                ShadowDepth = 0,
                BlurRadius = 15,
                Opacity = 0.7
            };

            window.SebastianHologram.Effect = shimmerEffect;
        }
    }
}