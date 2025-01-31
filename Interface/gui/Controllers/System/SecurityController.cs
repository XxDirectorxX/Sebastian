using System;
using System.Collections.ObjectModel;

namespace Sebastian.Controllers.System
{
    public class SecurityController
    {
        private readonly float FIELD_STRENGTH = 46.97871376f;
        private readonly float REALITY_COHERENCE = 1.618033988749895f;
        
        private readonly QuantumSystemBridge quantumBridge;
        private readonly ObservableCollection<SecurityZone> securityZones;
        
        public SecurityController()
        {
            quantumBridge = new QuantumSystemBridge();
            securityZones = new ObservableCollection<SecurityZone>();
            
            InitializeQuantumSystems();
            InitializeSecurityZones();
        }
        
        private void InitializeQuantumSystems()
        {
            float fieldEffect = FIELD_STRENGTH * REALITY_COHERENCE;
            quantumBridge.Initialize(fieldEffect);
        }
        
        private void InitializeSecurityZones()
        {
            securityZones.Add(new SecurityZone { Name = "Perimeter Defense", IsActive = true });
            securityZones.Add(new SecurityZone { Name = "Internal Monitoring", IsActive = true });
            securityZones.Add(new SecurityZone { Name = "Access Control", IsActive = true });
            securityZones.Add(new SecurityZone { Name = "Quantum Barrier", IsActive = true });
        }
        
        public void Initialize(float fieldStrength)
        {
            quantumBridge.UpdateQuantumState(fieldStrength);
        }
        
        public ObservableCollection<SecurityZone> GetSecurityZones()
        {
            return securityZones;
        }
        
        public void InitiateLockDown()
        {
            foreach (var zone in securityZones)
            {
                zone.IsActive = true;
            }
            
            float lockdownStrength = FIELD_STRENGTH;
            quantumBridge.UpdateFieldStrength(lockdownStrength);
        }
        
        public void ResetSystems()
        {
            float baseStrength = FIELD_STRENGTH * 0.5f;
            quantumBridge.UpdateFieldStrength(baseStrength);
        }
    }

    public class SecurityZone
    {
        public string Name { get; set; }
        public bool IsActive { get; set; }
    }
}
