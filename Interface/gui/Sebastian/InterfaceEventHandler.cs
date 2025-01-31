using System;
using System.Windows;
using System.Windows.Input;

namespace Sebastian
{
    public class InterfaceEventHandler
    {
        private readonly MainWindow window;
        private readonly AnimationController animationController;
        private readonly QuantumFieldVisualizer fieldVisualizer;
        private readonly ParticleSystem particleSystem;

        public InterfaceEventHandler(MainWindow mainWindow)
        {
            window = mainWindow;
            animationController = new AnimationController(window);
            fieldVisualizer = new QuantumFieldVisualizer(window.EnvironmentLayer);
            particleSystem = new ParticleSystem(window.ParticleSystem, window.EnvironmentLayer);

            InitializeEventHandlers();
        }

        private void InitializeEventHandlers()
        {
            window.MouseMove += OnMouseMove;
            window.KeyDown += OnKeyDown;
            window.Loaded += OnWindowLoaded;
        }

        private void OnMouseMove(object sender, MouseEventArgs e)
        {
            var position = e.GetPosition(window);
            UpdateQuantumField(position);
        }

        private void OnKeyDown(object sender, KeyEventArgs e)
        {
            switch (e.Key)
            {
                case Key.Escape:
                    window.Close();
                    break;
                case Key.F:
                    ToggleFullscreen();
                    break;
            }
        }

        private void OnWindowLoaded(object sender, RoutedEventArgs e)
        {
            particleSystem.Start();
        }

        private void UpdateQuantumField(Point position)
        {
            // Dynamic field updates based on mouse position
        }

        private void ToggleFullscreen()
        {
            window.WindowState = window.WindowState == WindowState.Normal 
                ? WindowState.Maximized 
                : WindowState.Normal;
        }
    }
}
