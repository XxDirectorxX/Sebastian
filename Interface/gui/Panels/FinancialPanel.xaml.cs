using System.Windows.Controls;
using Sebastian.Core;

namespace Sebastian.Panels
{
    public partial class FinancialPanel : UserControl
    {
        private readonly FinancialManager financialManager;
        private readonly QuantumSystemBridge quantumBridge;

        public FinancialPanel()
        {
            InitializeComponent();
            
            financialManager = new FinancialManager(FinancialDisplay);
            quantumBridge = new QuantumSystemBridge();
            
            InitializeQuantumComponents();
            SetupFinancialDisplay();
        }

        private void InitializeQuantumComponents()
        {
            double fieldEffect = Constants.FIELD_STRENGTH * Constants.REALITY_COHERENCE;
            financialManager.Initialize(fieldEffect);
            quantumBridge.Initialize(fieldEffect);
        }

        private void SetupFinancialDisplay()
        {
            ReportButton.Click += (s, e) => 
            {
                financialManager.GenerateReport();
                quantumBridge.UpdateFieldStrength(Constants.FIELD_STRENGTH);
            };
        }
    }
}
