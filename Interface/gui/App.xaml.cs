using System.Windows;
using gui.Core;

namespace gui
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;

        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);
            InitializeServices();
            InitializeQuantumField();
            SynchronizeControllers();
        }

        private void InitializeServices()
        {
            var container = ServiceContainer.Instance;
            var systemInitializer = container.GetService<SystemInitializer>();
            systemInitializer.BeginSystemOperations();
        }

        private void InitializeQuantumField()
        {
            var container = ServiceContainer.Instance;
            var quantumIntegrator = container.GetService<QuantumSystemIntegrator>();
            var fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            quantumIntegrator.ProcessQuantumState(fieldEffect);
        }

        private void SynchronizeControllers()
        {
            var container = ServiceContainer.Instance;
            var orchestrator = container.GetService<ControllerOrchestrator>();
            orchestrator.ProcessControllerEvent("SystemStartup", FIELD_STRENGTH);
        }
    }
}
