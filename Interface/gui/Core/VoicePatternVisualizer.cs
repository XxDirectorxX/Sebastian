using System;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace Sebastian.Core
{
    public class VoicePatternVisualizer
    {
        private readonly Canvas canvas;
        private readonly Random random;
        private bool isVisualizing;

        public VoicePatternVisualizer(Canvas canvas)
        {
            this.canvas = canvas;
            this.random = new Random();
            this.isVisualizing = false;
        }

        public void Initialize(double fieldStrength)
        {
            SetupVisualization();
        }

        public void StartVisualization()
        {
            isVisualizing = true;
            CompositionTarget.Rendering += UpdateVisualization;
        }

        public void StopVisualization()
        {
            isVisualizing = false;
            CompositionTarget.Rendering -= UpdateVisualization;
        }

        private void UpdateVisualization(object sender, EventArgs e)
        {
            if (!isVisualizing) return;

            canvas.Children.Clear();
            DrawQuantumPatterns();
        }

        private void DrawQuantumPatterns()
        {
            double resonance = Constants.PHI * Constants.FIELD_STRENGTH;
            for (int i = 0; i < 10; i++)
            {
                var pattern = new Ellipse
                {
                    Width = resonance,
                    Height = resonance,
                    Fill = new SolidColorBrush(Color.FromRgb(139, 0, 0)),
                    Opacity = 0.5
                };

                Canvas.SetLeft(pattern, random.NextDouble() * canvas.ActualWidth);
                Canvas.SetTop(pattern, random.NextDouble() * canvas.ActualHeight);
                canvas.Children.Add(pattern);
            }
        }

        private void SetupVisualization()
        {
            canvas.Background = new SolidColorBrush(Color.FromRgb(30, 30, 30));
        }
    }
}
