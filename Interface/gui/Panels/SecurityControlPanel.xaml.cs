using System;
using System.Windows;
using System.Windows.Controls;

namespace Sebastian.Panels
{
    public partial class SecurityControlPanel : UserControl
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private readonly SecurityController securityController;
        private readonly QuantumSystemBridge quantumBridge;
        
        public SecurityControlPanel()
        {
            InitializeComponent();
            
            securityController = new SecurityController();
            quantumBridge = new QuantumSystemBridge();
            
            InitializeQuantumComponents();
            SetupSecurityElements();
        }
        
        private void InitializeQuantumComponents()
        {
            float fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            securityController.Initialize(fieldEffect);
            quantumBridge.Initialize(fieldEffect);
        }
        
        private void SetupSecurityElements()
        {
            SecurityZones.ItemsSource = securityController.GetSecurityZones();
            LockDownButton.Click += OnLockDown;
            ResetButton.Click += OnReset;
        }
        
        private void OnLockDown(object sender, RoutedEventArgs e)
        {
            securityController.InitiateLockDown();
            SecurityStatus.Text = "LOCKDOWN ACTIVE";
            UpdateQuantumField(1.0f);
        }
        
        private void OnReset(object sender, RoutedEventArgs e)
        {
            securityController.ResetSystems();
            SecurityStatus.Text = "ACTIVE - All Systems Operational";
            UpdateQuantumField(0.5f);
        }
        
        private void UpdateQuantumField(float intensity)
        {
            float fieldStrength = intensity * FIELD_STRENGTH;
            quantumBridge.UpdateFieldStrength(fieldStrength);
        }
    }
}
