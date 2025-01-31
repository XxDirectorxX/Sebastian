using System.Windows;
using gui.Core;
using gui.Controllers.Core;
using gui.Controllers.Panels;
using gui.Controllers.Specialized;

namespace gui
{
    public partial class MainWindow : Window
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        private readonly ServiceContainer container;

        public MainWindow()
        {
            InitializeComponent();
            container = ServiceContainer.Instance;
            InitializeInterface();
            ActivateQuantumSystems();
        }

        private void InitializeInterface()
        {
            var quantumEffects = container.GetService<QuantumEffects>();
            var powerController = container.GetService<PowerController>();
            
            InitializePanels();
            InitializeSpecializedSystems();
            SetupEventHandlers();
        }

        private void InitializePanels()
        {
            DashboardPanel.DataContext = container.GetService<DashboardController>();
            ControlCenterPanel.DataContext = container.GetService<ControlCenterController>();
            FinancialPanel.DataContext = container.GetService<FinancialController>();
            HealthPanel.DataContext = container.GetService<HealthController>();
            SecurityPanel.DataContext = container.GetService<SecurityController>();
            SmartHelperPanel.DataContext = container.GetService<SmartHelperController>();
            SocialFamilyPanel.DataContext = container.GetService<SocialFamilyController>();
            AdvancedAbilitiesPanel.DataContext = container.GetService<AdvancedAbilitiesController>();
            VoiceCommunicationPanel.DataContext = container.GetService<VoiceCommunicationController>();
        }

        private void InitializeSpecializedSystems()
        {
            var fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            container.GetService<ContractSealController>().ProcessSealEvent(fieldEffect);
            container.GetService<RealityWarpController>().ProcessWarpEvent(fieldEffect);
            container.GetService<TimeControlController>().ProcessTimeEvent(fieldEffect);
            container.GetService<DemonicPowerController>().ProcessPowerEvent(fieldEffect);
        }

        private void ActivateQuantumSystems()
        {
            var orchestrator = container.GetService<ControllerOrchestrator>();
            orchestrator.ProcessControllerEvent("InterfaceReady", FIELD_STRENGTH);
        }

        private void SetupEventHandlers()
        {
            // Event handlers for interface interactions
        }
    }
}