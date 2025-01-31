using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Shapes;
using System.Windows.Threading;

namespace WpfApp1
{
    public class ParticleSystem
    {
        private readonly ItemsControl particleContainer;
        private readonly Canvas canvas;
        private readonly List<Particle> particles;
        private readonly Random random;
        private readonly DispatcherTimer timer;
        private const int MaxParticles = 100;

        public ParticleSystem(ItemsControl container, Canvas environmentCanvas)
        {
            particleContainer = container;
            canvas = environmentCanvas;
            particles = new List<Particle>();
            random = new Random();

            timer = new DispatcherTimer
            {
                Interval = TimeSpan.FromMilliseconds(16)
            };
            timer.Tick += UpdateParticles;
        }

        public void Start()
        {
            CreateInitialParticles();
            timer.Start();
        }

        private void CreateInitialParticles()
        {
            for (int i = 0; i < MaxParticles; i++)
            {
                CreateParticle();
            }
            UpdateParticleVisuals();
        }

        private void CreateParticle()
        {
            var particle = new Particle
            {
                Position = new Point(random.NextDouble() * canvas.ActualWidth,
                                   random.NextDouble() * canvas.ActualHeight),
                Velocity = new Vector((random.NextDouble() - 0.5) * 2,
                                    (random.NextDouble() - 0.5) * 2),
                Life = 1.0,
                Size = random.NextDouble() * 3 + 1
            };
            particles.Add(particle);
        }

        private void UpdateParticles(object sender, EventArgs e)
        {
            for (int i = particles.Count - 1; i >= 0; i--)
            {
                var particle = particles[i];
                particle.Position += particle.Velocity;
                particle.Life -= 0.01;

                if (particle.Life <= 0 || IsOutOfBounds(particle.Position))
                {
                    particles.RemoveAt(i);
                    CreateParticle();
                }
            }
            UpdateParticleVisuals();
        }

        private void UpdateParticleVisuals()
        {
            var collection = new List<UIElement>();
            foreach (var particle in particles)
            {
                var ellipse = new Ellipse
                {
                    Width = particle.Size,
                    Height = particle.Size,
                    Fill = new SolidColorBrush(Color.FromRgb(74, 28, 64)), // #4A1C40
                    Opacity = particle.Life
                };

                Canvas.SetLeft(ellipse, particle.Position.X);
                Canvas.SetTop(ellipse, particle.Position.Y);
                collection.Add(ellipse);
            }
            particleContainer.ItemsSource = collection;
        }

        private bool IsOutOfBounds(Point position)
        {
            return position.X < 0 || position.X > canvas.ActualWidth ||
                   position.Y < 0 || position.Y > canvas.ActualHeight;
        }
    }

    public class Particle
    {
        public Point Position { get; set; }
        public Vector Velocity { get; set; }
        public double Life { get; set; }
        public double Size { get; set; }
    }
}
