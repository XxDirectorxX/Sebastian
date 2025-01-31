using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace Sebastian.Core
{
    /// <summary>
    /// Visualizes quantum field effects and resonance patterns
    /// </summary>
    public class FieldVisualizer
    {
        private const double BASE_OPACITY = 0.7;
        private readonly Canvas canvas;
        private readonly SolidColorBrush fieldBrush;

        public FieldVisualizer(Canvas canvas)
        {
            this.canvas = canvas;
            this.fieldBrush = new SolidColorBrush(Color.FromRgb(139, 0, 0));
        }

        public void Initialize(double fieldStrength)
        {
            SetupFieldVisualization();
            UpdateField(fieldStrength);
        }

        public void UpdateField(double fieldStrength)
        {
            double opacity = BASE_OPACITY * (fieldStrength / 46.97871376);
            fieldBrush.Opacity = Math.Min(opacity, 1.0);
            
            canvas.Background = new LinearGradientBrush(
                Color.FromRgb(30, 30, 30),
                fieldBrush.Color,
                new Point(0, 0),
                new Point(1, 1));
        }

        private void SetupFieldVisualization()
        {
            canvas.Background = new SolidColorBrush(Color.FromRgb(30, 30, 30));
        }
    }
}
