using System;
using System.Windows.Threading;

namespace Sebastian
{
    public class StatusMonitor
    {
        private readonly MainWindow window;
        private readonly DispatcherTimer timer;
        private readonly double fieldStrength = 46.97871376;
        private readonly double realityCoherence = 1.618033988749895;
        private readonly Random random = new Random();

        public StatusMonitor(MainWindow mainWindow)
        {
            window = mainWindow;
            timer = new DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(1)
            };
            timer.Tick += UpdateStatus;
        }

        public void Start()
        {
            timer.Start();
            UpdateStatus(null, null);
        }

        private void UpdateStatus(object sender, EventArgs e)
        {
            window.FieldStrengthDisplay.Text = $"Field Strength: {fieldStrength + GetQuantumFluctuation():F8}";
            window.CoherenceDisplay.Text = $"Reality Coherence: {realityCoherence + GetQuantumFluctuation():F12}";
            window.SystemStatusDisplay.Text = $"System Status: {GetSystemStatus()}";
        }

        private double GetQuantumFluctuation()
        {
            return (random.NextDouble() - 0.5) * 0.000001;
        }

        private string GetSystemStatus()
        {
            string[] states = { "Optimal", "Synchronized", "Quantum-Aligned", "Perfect" };
            return states[random.Next(states.Length)];
        }
    }
}
