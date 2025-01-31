using System;
using System.Windows;

namespace Sebastian
{
    public class Program
    {
        [STAThread]
        public static void Main()
        {
            var application = new Application();
            var mainWindow = new MainWindow();
            
            // Initialize quantum framework
            InitializeQuantumFramework();
            
            // Start application
            application.Run(mainWindow);
        }

        private static void InitializeQuantumFramework()
        {
            // Set core quantum parameters
            const double fieldStrength = 46.97871376;
            const double realityCoherence = 1.618033988749895;

            // Initialize quantum systems
            QuantumSystemManager.Initialize(fieldStrength, realityCoherence);
        }
    }
}
