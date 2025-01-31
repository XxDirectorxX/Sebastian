using System;
using System.Collections.Generic;
using gui.Controllers.Core;
using gui.Controllers.Panels;
using gui.Controllers.Specialized;

namespace gui.Core
{
    public class ServiceContainer
    {
        private static ServiceContainer instance;
        private readonly Dictionary<Type, object> services;

        public static ServiceContainer Instance
        {
            get
            {
                instance ??= new ServiceContainer();
                return instance;
            }
        }

        private ServiceContainer()
        {
            services = new Dictionary<Type, object>();
            RegisterServices();
        }

        private void RegisterServices()
        {
            // Core Controllers
            RegisterService<QuantumSystemIntegrator>();
            RegisterService<ControllerOrchestrator>();
            RegisterService<SystemInitializer>();
            RegisterService<PowerController>();
            RegisterService<QuantumEffects>();

            // Panel Controllers
            RegisterService<DashboardController>();
            RegisterService<ControlCenterController>();
            RegisterService<FinancialController>();
            RegisterService<HealthController>();
            RegisterService<SecurityController>();
            RegisterService<SmartHelperController>();
            RegisterService<SocialFamilyController>();
            RegisterService<AdvancedAbilitiesController>();
            RegisterService<VoiceCommunicationController>();
            RegisterService<DemonicController>();

            // Specialized Controllers
            RegisterService<ContractSealController>();
            RegisterService<PowerManagementController>();
            RegisterService<RealityWarpController>();
            RegisterService<TimeControlController>();
            RegisterService<EnergyProjectionController>();
            RegisterService<QuantumBridgeController>();
            RegisterService<RealityManipulationController>();
            RegisterService<TimeManipulationController>();
            RegisterService<VoiceSynthesisController>();
            RegisterService<DialogueController>();
            RegisterService<BehaviorController>();
            RegisterService<PersonalityController>();
            RegisterService<CombatController>();
            RegisterService<ButlerController>();
            RegisterService<EleganceController>();
            RegisterService<LoyaltyController>();
            RegisterService<EfficiencyController>();
            RegisterService<DemonicPowerController>();
            RegisterService<QuantumFieldController>();
            RegisterService<SettingsController>();
        }

        private void RegisterService<T>() where T : new()
        {
            services[typeof(T)] = new T();
        }

        public T GetService<T>()
        {
            return (T)services[typeof(T)];
        }
    }
}
