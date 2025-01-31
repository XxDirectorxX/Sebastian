using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace Sebastian.Core
{
    /// <summary>
    /// Manages quantum particle visualization and behavior
    /// </summary>
    public class ParticleSystem
    {
        private const int MAX_PARTICLES = 100;
        private readonly Random random;
        private readonly List<Particle> particles;
        private readonly Canvas canvas;
        private readonly ItemsControl particleContainer;

        public ParticleSystem(Canvas canvas, ItemsControl container)
        {
            this.canvas = canvas;
            this.particleContainer = container;
            this.random = new Random();
            this.particles = new List<Particle>();
        }

        public void Initialize(double fieldStrength)
        {
            CreateInitialParticles();
            StartParticleSystem();
        }

        public void UpdateIntensity(double intensity)
        {
            foreach (var particle in particles)
            {
                particle.UpdateVelocity(intensity);
            }
        }

        private void CreateInitialParticles()
        {
            for (int i = 0; i < MAX_PARTICLES; i++)
            {
                CreateParticle();
            }
            UpdateParticleVisuals();
        }

        private void StartParticleSystem()
        {
            CompositionTarget.Rendering += UpdateParticles;
        }

        private void UpdateParticles(object sender, EventArgs e)
        {
            UpdateParticlePositions();
            UpdateParticleVisuals();
        }
    }
}
