using System;
using System.Windows;
using System.Windows.Controls;

namespace Sebastian.Panels
{
    public partial class DashboardPanel : UserControl
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private readonly DashboardController dashboardController;
        private readonly QuantumSystemBridge quantumBridge;
        private readonly ParticleSystem particleSystem;
        private readonly StatusMonitor statusMonitor;
        
        public DashboardPanel()
        {
            InitializeComponent();
            
            dashboardController = new DashboardController();
            quantumBridge = new QuantumSystemBridge();
            particleSystem = new ParticleSystem(QuantumFieldCanvas);
            statusMonitor = new StatusMonitor(StatusList);
            
            InitializeQuantumComponents();
            SetupDashboardElements();
            ActivateDashboardSystems();
        }
        
        private void InitializeQuantumComponents()
        {
            float fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            dashboardController.Initialize(fieldEffect);
            quantumBridge.Initialize(fieldEffect);
            particleSystem.Initialize(fieldEffect);
            statusMonitor.Initialize(fieldEffect);
        }
        
        private void SetupDashboardElements()
        {
            SyncButton.Click += OnSync;
            OptimizeButton.Click += OnOptimize;
            
            InitializeTaskList();
            InitializeHealthMonitor();
        }
        
        private void ActivateDashboardSystems()
        {
            dashboardController.ActivateSystems();
            particleSystem.Start();
            statusMonitor.StartMonitoring();
        }
        
        private void InitializeTaskList()
        {
            TaskList.ItemsSource = dashboardController.GetActiveTasks();
        }
        
        private void InitializeHealthMonitor()
        {
            dashboardController.InitializeHealthMonitoring(HealthCanvas);
        }
        
        private void OnSync(object sender, RoutedEventArgs e)
        {
            float syncCoherence = REALITY_COHERENCE;
            dashboardController.SynchronizeSystems(syncCoherence);
            quantumBridge.SynchronizeField(syncCoherence);
            particleSystem.Synchronize();
        }
        
        private void OnOptimize(object sender, RoutedEventArgs e)
        {
            float optimizationStrength = FIELD_STRENGTH;
            dashboardController.OptimizeSystems(optimizationStrength);
            quantumBridge.OptimizeField(optimizationStrength);
            particleSystem.Optimize();
        }
    }
}
