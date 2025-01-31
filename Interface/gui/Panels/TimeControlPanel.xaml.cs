using System;
using System.Windows;
using System.Windows.Controls;

namespace Sebastian.Panels
{
    public partial class TimeControlPanel : UserControl
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private readonly TimeControlController timeController;
        private readonly QuantumSystemBridge quantumBridge;
        private readonly ParticleSystem particleSystem;
        
        public TimeControlPanel()
        {
            InitializeComponent();
            
            timeController = new TimeControlController();
            quantumBridge = new QuantumSystemBridge();
            particleSystem = new ParticleSystem(TimeCanvas);
            
            InitializeQuantumComponents();
            SetupTimeElements();
            ActivateTimeSystems();
        }
        
        private void InitializeQuantumComponents()
        {
            float fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            timeController.ProcessTime(fieldEffect);
            quantumBridge.Initialize(fieldEffect);
            particleSystem.Initialize(fieldEffect);
        }
        
        private void SetupTimeElements()
        {
            FlowSlider.ValueChanged += OnFlowChanged;
            DilationSlider.ValueChanged += OnDilationChanged;
            SyncSlider.ValueChanged += OnSyncChanged;
            
            ManipulateButton.Click += OnManipulate;
            ResetButton.Click += OnReset;
        }
        
        private void ActivateTimeSystems()
        {
            timeController.UpdateTime(FIELD_STRENGTH);
            quantumBridge.SynchronizeField(REALITY_COHERENCE);
            particleSystem.Start();
        }
        
        private void OnFlowChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            float flow = (float)e.NewValue / 100f * FIELD_STRENGTH;
            timeController.UpdateFlow(flow);
        }
        
        private void OnDilationChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            float dilation = (float)e.NewValue / 100f * FIELD_STRENGTH;
            timeController.UpdateDilation(dilation);
        }
        
        private void OnSyncChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            float sync = (float)e.NewValue / 100f * REALITY_COHERENCE;
            timeController.UpdateSync(sync);
        }
        
        private void OnManipulate(object sender, RoutedEventArgs e)
        {
            timeController.Manipulate();
            particleSystem.Activate();
        }
        
        private void OnReset(object sender, RoutedEventArgs e)
        {
            quantumBridge.SynchronizeField(REALITY_COHERENCE);
            particleSystem.Reset();
        }
    }
}
