using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Shapes;

namespace WpfApp1
{
    public class QuantumFieldVisualizer
    {
        private readonly Canvas canvas;
        private readonly Path fieldPath;
        private readonly double fieldStrength = 46.97871376;
        private readonly double phi = 1.618033988749895;
        private readonly Random random = new Random();

        public QuantumFieldVisualizer(Canvas environmentCanvas)
        {
            canvas = environmentCanvas;
            fieldPath = new Path
            {
                Stroke = new SolidColorBrush(Color.FromRgb(191, 155, 48)), // #BF9B30
                StrokeThickness = 1,
                Opacity = 0.6
            };
            canvas.Children.Add(fieldPath);
        }

        public void Start()
        {
            var animation = new DoubleAnimation
            {
                From = 0,
                To = 360,
                Duration = TimeSpan.FromSeconds(30),
                RepeatBehavior = RepeatBehavior.Forever
            };

            fieldPath.RenderTransform = new RotateTransform();
            fieldPath.RenderTransform.BeginAnimation(RotateTransform.AngleProperty, animation);
            
            CompositionTarget.Rendering += UpdateField;
        }

        private void UpdateField(object sender, EventArgs e)
        {
            var geometry = new StreamGeometry();
            using (var context = geometry.Open())
            {
                DrawQuantumField(context);
            }
            geometry.Freeze();
            fieldPath.Data = geometry;
        }

        private void DrawQuantumField(StreamGeometryContext context)
        {
            const int points = 360;
            const double radius = 200;

            for (int i = 0; i < points; i++)
            {
                double angle = i * Math.PI / 180;
                double variation = Math.Sin(angle * phi) * 20;
                double r = radius + variation;
                
                double x = Math.Cos(angle) * r + canvas.ActualWidth / 2;
                double y = Math.Sin(angle) * r + canvas.ActualHeight / 2;

                if (i == 0)
                    context.BeginFigure(new Point(x, y), false, false);
                else
                    context.LineTo(new Point(x, y), true, false);
            }
        }
    }
}