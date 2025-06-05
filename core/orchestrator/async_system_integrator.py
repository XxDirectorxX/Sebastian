"""
Asynchronous system integrator for Sebastian assistant.

Connects all major subsystems and manages dataflow between components with non-blocking operations.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class AsyncSystemIntegrator:
    """
    Asynchronous integration of all Sebastian assistant subsystems:
    - Intelligence (NLP, intent parsing)
    - Memory (short-term, long-term)
    - Personality (dialogue, tone, mannerisms)
    - Orchestrator (task dispatch, plugins)
    - I/O systems (voice, vision)
    - Learning (adaptive behavior)
    
    Implements non-blocking operations for responsive interaction.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize system integrator with all subsystems.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._loop = asyncio.get_event_loop()
        self._processing_tasks = set()
        self._shutdown_event = asyncio.Event()
        
        # Import all required components
        from core.intelligence.intent_parser import IntentParser
        from core.intelligence.sentiment_analysis import SentimentAnalyzer
        from core.intelligence.emotion_detector import EmotionDetector
        from core.memory.memory_interface import MemoryInterface
        from core.knowledge.knowledge_base import KnowledgeBase
        from core.personality_simulation import PersonalitySimulator
        from core.orchestrator.task_dispatcher import TaskDispatcher
        from core.orchestrator.plugin_orchestrator import get_plugin_orchestrator
        from core.context_manager import ContextManager
        from core.learning.learning_algorithms import LearningAlgorithms
        
        # Initialize subsystems
        logger.info("Initializing subsystems...")
        
        self.intent_parser = IntentParser()
        logger.info("Intent parser initialized")
        
        self.sentiment_analyzer = SentimentAnalyzer()
        logger.info("Sentiment analyzer initialized")
        
        self.emotion_detector = EmotionDetector()
        logger.info("Emotion detector initialized")
        
        self.memory = MemoryInterface()
        logger.info("Memory interface initialized")
        
        self.knowledge_base = KnowledgeBase()
        logger.info("Knowledge base initialized")
        
        # Initialize learning system with memory access
        self.learning = LearningAlgorithms(memory_interface=self.memory)
        logger.info("Learning algorithms initialized")
        
        self.personality = PersonalitySimulator()
        logger.info("Personality simulator initialized")
        
        self.task_dispatcher = TaskDispatcher(self.intent_parser)
        logger.info("Task dispatcher initialized")
        
        self.plugin_orchestrator = get_plugin_orchestrator()
        logger.info("Plugin orchestrator initialized")
        
        self.context_manager = ContextManager()
        logger.info("Context manager initialized")
        
        # System state
        self.active_tasks = {}
        self.system_health = {"status": "healthy", "subsystems": {}}
        
        logger.info("Asynchronous system integrator initialized successfully")
    
    async def initialize(self):
        """Initialize asynchronous components and resources."""
        # Perform async initialization tasks
        await self.memory.async_init()
        await self.knowledge_base.async_init()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        maintenance_task = asyncio.create_task(self._periodic_maintenance())
        self._processing_tasks.add(maintenance_task)
        maintenance_task.add_done_callback(self._processing_tasks.discard)
    
    async def _periodic_maintenance(self):
        """Perform periodic system maintenance."""
        try:
            while not self._shutdown_event.is_set():
                # Check system health
                await self._check_subsystem_health()
                
                # Run memory optimization
                await self.memory.optimize()
                
                # Update learning models
                await asyncio.to_thread(self.learning.periodic_update)
                
                # Wait for next cycle
                await asyncio.sleep(300)  # 5 minutes
        except asyncio.CancelledError:
            logger.info("Maintenance task cancelled")
        except Exception as e:
            logger.error(f"Error in maintenance task: {e}")
    
    async def _check_subsystem_health(self):
        """Check health of all subsystems."""
        # Implement health checks for each subsystem
        subsystems = {
            "memory": self.memory,
            "knowledge": self.knowledge_base,
            "personality": self.personality,
            "plugins": self.plugin_orchestrator
        }
        
        for name, subsystem in subsystems.items():
            try:
                if hasattr(subsystem, "health_check"):
                    status = await subsystem.health_check()
                    self.system_health["subsystems"][name] = status
                else:
                    self.system_health["subsystems"][name] = {"status": "unknown"}
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                self.system_health["subsystems"][name] = {"status": "error", "error": str(e)}
        
        # Overall system status
        errors = [s for s in self.system_health["subsystems"].values() if s.get("status") == "error"]
        if errors:
            self.system_health["status"] = "degraded"
        else:
            self.system_health["status"] = "healthy"
    
    async def process_input(self, input_text: str, input_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process user input through all subsystems to generate response asynchronously.
        
        Args:
            input_text: User input text
            input_context: Additional context information
            
        Returns:
            Response with generated text and metadata
        """
        context = input_context or {}
        user_id = context.get("user_id", "default")
        processing_id = context.get("processing_id", str(id(input_text)))
        
        try:
            # Start tracking this task
            self.active_tasks[processing_id] = {
                "status": "processing",
                "start_time": datetime.datetime.now().isoformat(),
                "user_id": user_id,
                "text": input_text[:50] + ("..." if len(input_text) > 50 else "")
            }
            
            # Update context
            self.context_manager.update_context("last_input", input_text)
            for key, value in context.items():
                self.context_manager.update_context(key, value)
            
            # Create processing tasks
            intent_task = asyncio.create_task(self._parse_intent(input_text))
            emotion_task = asyncio.create_task(self._analyze_emotion(input_text))
            memory_task = asyncio.create_task(self._retrieve_memories(input_text))
            knowledge_task = asyncio.create_task(self._retrieve_knowledge(input_text))
            
            # Wait for initial processing tasks
            intent_result, emotion_result, memories, knowledge = await asyncio.gather(
                intent_task, emotion_task, memory_task, knowledge_task
            )
            
            # Update context with results
            intent = intent_result["intent"]
            self.context_manager.update_context("intent", intent)
            self.context_manager.update_context("emotion", emotion_result)
            self.context_manager.update_context("recent_memories", memories["recent"])
            self.context_manager.update_context("related_memories", memories["related"])
            self.context_manager.update_context("relevant_facts", knowledge)
            
            # Update memory asynchronously (don't await)
            memory_update_task = asyncio.create_task(self.memory.async_remember(
                f"User: {input_text}", 
                {
                    "type": "user_input",
                    "intent": intent,
                    "emotion": emotion_result,
                    "timestamp": context.get("timestamp", datetime.datetime.now().isoformat()),
                    "user_id": user_id
                }
            ))
            
            # Get learning-based adaptations (CPU-bound, use thread)
            learning_params = await asyncio.to_thread(
                self.learning.update_from_interaction,
                user_id=user_id,
                interaction_data={
                    "text": input_text,
                    "intent": intent,
                    "timestamp": context.get("timestamp", datetime.datetime.now().isoformat())
                }
            )
            
            # Apply learning-based adaptations to context
            self.context_manager.update_context("formality_level", learning_params["formality_level"])
            self.context_manager.update_context("user_preferences", learning_params["user_preferences"])
            
            # Execute task via appropriate plugin (may involve external API calls)
            try:
                plugin_context = {
                    "user_context": context,
                    "system_context": self.context_manager.get_context(),
                    "memory": self.memory,
                    "knowledge": self.knowledge_base
                }
                plugin_result = await self._execute_plugin(intent, intent_result, plugin_context)
            except Exception as e:
                logger.error(f"Plugin execution error: {e}")
                plugin_result = {
                    "success": False,
                    "error": str(e),
                    "message": "I apologize, but I encountered an error processing your request."
                }
            
            # Format response with personality (CPU-bound, use thread)
            raw_response = plugin_result.get("message", "I processed your request.")
            
            # Check if there are recommended responses from learning
            recommended_responses = learning_params.get("recommended_responses", [])
            if recommended_responses and intent in plugin_result.get("adaptable_intents", []):
                # Use a recommended response pattern if available
                raw_response = recommended_responses[0]
            
            response_data = {
                "content": raw_response,
                "intent": intent,
                "emotion": emotion_result
            }
            
            # Apply personality with adaptive formality level
            full_context = self.context_manager.get_context()
            formatted_response = await asyncio.to_thread(
                self.personality.dialogue_manager.format_response,
                response_data,
                full_context
            )
            
            # Update memory with response asynchronously (don't await)
            response_memory_task = asyncio.create_task(self.memory.async_remember(
                f"Sebastian: {formatted_response}",
                {
                    "type": "assistant_response",
                    "intent": intent,
                    "user_id": user_id
                }
            ))
            
            # Record interaction for learning asynchronously
            learning_update_task = asyncio.create_task(asyncio.to_thread(
                self.learning.update_from_interaction,
                user_id=user_id,
                interaction_data={
                    "text": input_text,
                    "response": formatted_response,
                    "intent": intent,
                    "timestamp": context.get("timestamp", datetime.datetime.now().isoformat())
                }
            ))
            
            # Prepare final result
            result = {
                "text": formatted_response,
                "intent": intent,
                "emotion": emotion_result,
                "plugin_data": plugin_result,
                "success": plugin_result.get("success", True),
                "processing_id": processing_id,
                "learning": {
                    "formality_level": learning_params["formality_level"],
                    "has_preferences": bool(learning_params["user_preferences"])
                }
            }
            
            # Mark task as completed
            self.active_tasks[processing_id]["status"] = "completed"
            self.active_tasks[processing_id]["end_time"] = datetime.datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing input: {e}", exc_info=True)
            # Mark task as failed
            if processing_id in self.active_tasks:
                self.active_tasks[processing_id]["status"] = "failed"
                self.active_tasks[processing_id]["error"] = str(e)
                self.active_tasks[processing_id]["end_time"] = datetime.datetime.now().isoformat()
            
            # Return graceful error response
            return {
                "text": "I do apologize, but I've encountered an unexpected issue while processing your request. Rest assured, I shall address this matter promptly.",
                "error": str(e),
                "success": False,
                "processing_id": processing_id
            }
    
    async def _parse_intent(self, text: str) -> Dict[str, Any]:
        """Parse intent from text."""
        return await asyncio.to_thread(self.intent_parser.parse, text)
    
    async def _analyze_emotion(self, text: str) -> Dict[str, Any]:
        """Analyze emotion in text."""
        return await asyncio.to_thread(self.emotion_detector.analyze_text, text)
    
    async def _retrieve_memories(self, text: str) -> Dict[str, List]:
        """Retrieve relevant memories."""
        recent = await self.memory.async_recall_recent(5)
        related = await self.memory.async_query_knowledge(text, 3)
        return {"recent": recent, "related": related}
    
    async def _retrieve_knowledge(self, text: str) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge."""
        return await self.knowledge_base.async_query_relevant(text)
    
    async def _execute_plugin(self, intent: str, parsed_data: Dict, context: Dict) -> Dict:
        """Execute plugin for intent."""
        if hasattr(self.plugin_orchestrator, "async_execute_plugin"):
            return await self.plugin_orchestrator.async_execute_plugin(intent, parsed_data, context)
        else:
            # Fall back to synchronous execution if async not implemented
            return await asyncio.to_thread(self.plugin_orchestrator.execute_plugin, intent, parsed_data)
    
    async def shutdown(self):
        """Gracefully shut down the system."""
        logger.info("Initiating system shutdown...")
        
        # Signal shutdown to background tasks
        self._shutdown_event.set()
        
        # Wait for all tasks to complete
        if self._processing_tasks:
            logger.info(f"Waiting for {len(self._processing_tasks)} tasks to complete...")
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)
        
        # Close executor
        self._executor.shutdown(wait=True)
        
        # Close resources
        await self.memory.async_close()
        await self.knowledge_base.async_close()
        
        logger.info("System shutdown complete")

# Create singleton instance
_async_system_integrator = None

async def get_async_system_integrator(config: Optional[Dict[str, Any]] = None):
    """Get singleton instance of AsyncSystemIntegrator."""
    global _async_system_integrator
    if _async_system_integrator is None:
        _async_system_integrator = AsyncSystemIntegrator(config)
        await _async_system_integrator.initialize()
    return _async_system_integrator