"""
System integrator for Sebastian assistant.

Connects all major subsystems and manages dataflow between components.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class SubsystemError(Exception):
    pass

class SystemIntegrator:
    """
    Integrates all subsystems of the Sebastian assistant:
    - Intelligence (NLP, intent parsing)
    - Memory (short-term, long-term)
    - Personality (dialogue, tone, mannerisms)
    - Orchestrator (task dispatch, plugins)
    - I/O systems (voice, vision)
    - Learning (adaptive behavior)
    - Security (authentication, encryption)
    
    Serves as the central nervous system that coordinates dataflow.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, persona_say = None):
        """
        Initialize system integrator with all subsystems.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.persona_say = persona_say
        self.voice = None
        self.vision = None
        self.memory = None
        self.initialized = False
        
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
        from core.security.security_manager import SecurityManager
        
        # Initialize subsystems
        logger.info("Initializing subsystems...")
        
        # Initialize security first
        self.security_manager = SecurityManager(self.config.get("security", {}))
        logger.info("Security manager initialized")
        
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
        self.learning = LearningAlgorithms(
            memory_interface=self.memory,
            knowledge_base=self.knowledge_base
        )
        logger.info("Learning algorithms initialized")
        
        self.personality = PersonalitySimulator()
        logger.info("Personality simulator initialized")
        
        self.task_dispatcher = TaskDispatcher(self.intent_parser)
        logger.info("Task dispatcher initialized")
        
        self.plugin_orchestrator = get_plugin_orchestrator()
        logger.info("Plugin orchestrator initialized")
        
        self.context_manager = ContextManager()
        logger.info("Context manager initialized")
        
        # Task queue for asynchronous processing
        self.task_queue = asyncio.Queue()
        self.processing_tasks = []
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("All subsystems initialized successfully")
        
    async def initialize(self):
        self.persona_say("Commencing subsystem initialization, My Lord.")
        try:
            await self._init_voice()
            await self._init_vision()
            await self._init_memory()
            self.initialized = True
            self.persona_say("All subsystems are now operational, My Lord.")
        except Exception as e:
            self.logger.exception("Subsystem initialization failed.")
            self.persona_say(f"A subsystem has failed to initialize: {e}", error=True)
            raise SubsystemError from e

    async def _init_voice(self):
        if self.config.get("voice", {}).get("enabled", False):
            try:
                # Placeholder for actual voice subsystem import and init
                await asyncio.sleep(0.05)
                self.voice = "VoiceSubsystem"
                self.persona_say("Voice subsystem is ready to serve.")
            except Exception as e:
                self.persona_say("Voice subsystem failed to initialize.", error=True)
                raise

    async def _init_vision(self):
        if self.config.get("vision", {}).get("enabled", False):
            try:
                # Placeholder for actual vision subsystem import and init
                await asyncio.sleep(0.05)
                self.vision = "VisionSubsystem"
                self.persona_say("Vision subsystem is at your disposal.")
            except Exception as e:
                self.persona_say("Vision subsystem failed to initialize.", error=True)
                raise

    async def _init_memory(self):
        try:
            # Placeholder for actual memory subsystem import and init
            await asyncio.sleep(0.05)
            self.memory = "MemorySubsystem"
            self.persona_say("Memory subsystem is fully operational.")
        except Exception as e:
            self.persona_say("Memory subsystem failed to initialize.", error=True)
            raise

    def _start_background_tasks(self):
        """Start background tasks for asynchronous processing."""
        # Create task worker coroutine
        async def task_worker():
            while True:
                try:
                    task_info = await self.task_queue.get()
                    task_type = task_info.get("type")
                    task_data = task_info.get("data", {})
                    
                    if task_type == "shutdown":
                        break
                        
                    # Process based on task type
                    if task_type == "memory_consolidation":
                        await self._process_memory_consolidation()
                    elif task_type == "knowledge_update":
                        await self._process_knowledge_update(task_data)
                    elif task_type == "learning_update":
                        await self._process_learning_update(task_data)
                    
                    self.task_queue.task_done()
                except Exception as e:
                    logger.error(f"Error in background task: {e}")
        
        # Start the worker tasks
        for i in range(3):  # 3 worker tasks
            task = asyncio.create_task(task_worker())
            self.processing_tasks.append(task)
            
    async def _process_memory_consolidation(self):
        """Process memory consolidation in background."""
        try:
            logger.debug("Running memory consolidation")
            # This would call memory consolidation methods
            # For now, just a placeholder
            await asyncio.sleep(0.1)  # Simulate some work
        except Exception as e:
            logger.error(f"Memory consolidation error: {e}")
            
    async def _process_knowledge_update(self, data: Dict[str, Any]):
        """Process knowledge base update in background."""
        try:
            logger.debug(f"Updating knowledge base: {data.get('topic')}")
            # This would update the knowledge base
            # For now, just a placeholder
            await asyncio.sleep(0.1)  # Simulate some work
        except Exception as e:
            logger.error(f"Knowledge update error: {e}")
            
    async def _process_learning_update(self, data: Dict[str, Any]):
        """Process learning update in background."""
        try:
            logger.debug("Updating learning models")
            # This would update learning models
            # For now, just a placeholder
            await asyncio.sleep(0.1)  # Simulate some work
        except Exception as e:
            logger.error(f"Learning update error: {e}")
        
    async def process_input_async(self, input_text: str, input_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process user input asynchronously through all subsystems to generate response.
        
        Args:
            input_text: User input text
            input_context: Additional context information
            
        Returns:
            Response with generated text and metadata
        """
        context = input_context or {}
        user_id = context.get("user_id", "default")
        
        # Security: Sanitize input
        sanitized_input = self.security_manager.sanitize_input(input_text)
        
        # Update context
        self.context_manager.update_context("last_input", sanitized_input)
        for key, value in context.items():
            self.context_manager.update_context(key, value)
            
        # Parse intent
        logger.debug(f"Parsing intent: {sanitized_input}")
        parsed = self.intent_parser.parse(sanitized_input)
        intent = parsed["intent"]
        
        # Analyze emotion (more comprehensive than just sentiment)
        logger.debug("Analyzing emotion")
        emotion_data = self.emotion_detector.analyze_text(sanitized_input)
        parsed["emotion"] = emotion_data
        
        # Update memory
        logger.debug("Updating memory")
        await self._update_memory(sanitized_input, user_id, intent, emotion_data, context)
        
        # Get learning-based adaptations for this user
        learning_params = await self._get_learning_params(user_id, sanitized_input, intent, context)
        
        # Apply learning-based adaptations to context
        self.context_manager.update_context("formality_level", learning_params["formality_level"])
        self.context_manager.update_context("user_preferences", learning_params["user_preferences"])
        
        # Retrieve relevant knowledge and memories
        knowledge_context = await self._gather_knowledge_context(sanitized_input)
        
        # Update context with knowledge information
        self.context_manager.update_context("relevant_facts", knowledge_context["relevant_facts"])
        self.context_manager.update_context("recent_memories", knowledge_context["recent_memories"])
        self.context_manager.update_context("related_memories", knowledge_context["related_memories"])
        
        # Execute task via appropriate plugin
        logger.debug(f"Dispatching task: {intent}")
        try:
            plugin_context = {
                "user_context": context,
                "system_context": self.context_manager.get_context(),
                "memory": self.memory,
                "knowledge": self.knowledge_base,
                "security": self.security_manager
            }
            
            # Check access permission if security is enabled
            if context.get("session_token") and not self.security_manager.check_access(
                context["session_token"], "plugin", intent
            ):
                plugin_result = {
                    "success": False,
                    "error": "access_denied",
                    "message": "I apologize, but you don't have permission to perform this action."
                }
            else:
                # Execute plugin with timeout protection
                plugin_result = await asyncio.wait_for(
                    self._execute_plugin(intent, parsed, plugin_context),
                    timeout=10.0  # 10 second timeout
                )
                
        except asyncio.TimeoutError:
            logger.error(f"Plugin execution timed out for intent: {intent}")
            plugin_result = {
                "success": False,
                "error": "timeout",
                "message": "I apologize, but processing your request is taking longer than expected."
            }
        except Exception as e:
            logger.error(f"Plugin execution error: {e}", exc_info=True)
            plugin_result = {
                "success": False,
                "error": str(e),
                "message": "I apologize, but I encountered an error processing your request."
            }
        
        # Format response with personality
        logger.debug("Applying personality")
        raw_response = plugin_result.get("message", "I processed your request.")
        
        # Check if there are recommended responses from learning
        recommended_responses = learning_params.get("recommended_responses", [])
        if recommended_responses and intent in plugin_result.get("adaptable_intents", []):
            # Use a recommended response pattern if available
            raw_response = recommended_responses[0]
        
        response_data = {
            "content": raw_response,
            "intent": intent,
            "emotion": emotion_data
        }
        
        # Apply personality with adaptive formality level
        full_context = self.context_manager.get_context()
        formatted_response = self.personality.shape_response(
            response_data,
            full_context
        )
        
        # Update memory with response
        await self._update_response_memory(formatted_response, user_id, intent)
        
        # Schedule learning update in background
        self.task_queue.put_nowait({
            "type": "learning_update",
            "data": {
                "user_id": user_id,
                "input_text": sanitized_input,
                "response": formatted_response,
                "intent": intent,
                "timestamp": context.get("timestamp")
            }
        })
        
        # Prepare final result
        result = {
            "text": formatted_response,
            "intent": intent,
            "emotion": emotion_data,
            "plugin_data": plugin_result,
            "success": plugin_result.get("success", True),
            "learning": {
                "formality_level": learning_params["formality_level"],
                "has_preferences": bool(learning_params["user_preferences"])
            }
        }
        
        return result
        
    async def _update_memory(self, input_text: str, user_id: str, intent: str, 
                           emotion_data: Dict[str, Any], context: Dict[str, Any]):
        """Update memory with user input."""
        memory_entry = f"User: {input_text}"
        memory_metadata = {
            "type": "user_input",
            "intent": intent,
            "emotion": emotion_data,
            "timestamp": context.get("timestamp", ""),
            "user_id": user_id
        }
        
        # Use async version if memory interface supports it
        if hasattr(self.memory, "remember_async"):
            await self.memory.remember_async(memory_entry, memory_metadata)
        else:
            self.memory.remember(memory_entry, memory_metadata)
            
    async def _update_response_memory(self, response: str, user_id: str, intent: str):
        """Update memory with assistant response."""
        memory_entry = f"Sebastian: {response}"
        memory_metadata = {
            "type": "assistant_response",
            "intent": intent,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Use async version if memory interface supports it
        if hasattr(self.memory, "remember_async"):
            await self.memory.remember_async(memory_entry, memory_metadata)
        else:
            self.memory.remember(memory_entry, memory_metadata)
            
    async def _get_learning_params(self, user_id: str, input_text: str, 
                                intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get learning parameters for user."""
        # Use async version if learning system supports it
        if hasattr(self.learning, "update_from_interaction_async"):
            return await self.learning.update_from_interaction_async(
                user_id=user_id,
                interaction_data={
                    "text": input_text,
                    "intent": intent,
                    "timestamp": context.get("timestamp")
                }
            )
        else:
            # Fall back to synchronous version
            return self.learning.update_from_interaction(
                user_id=user_id,
                interaction_data={
                    "text": input_text,
                    "intent": intent,
                    "timestamp": context.get("timestamp")
                }
            )
            
    async def _gather_knowledge_context(self, input_text: str) -> Dict[str, List]:
        """Gather relevant knowledge and memories."""
        # These could be async calls in the future
        recent_memories = self.memory.recall_recent(5)
        related_memories = self.memory.query_knowledge(input_text, 3)
        relevant_facts = self.knowledge_base.query_relevant(input_text)
        
        return {
            "recent_memories": recent_memories,
            "related_memories": related_memories,
            "relevant_facts": relevant_facts
        }
            
    async def _execute_plugin(self, intent: str, parsed_data: Dict[str, Any], 
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute plugin with async support."""
        # Check if plugin supports async
        plugin = self.plugin_orchestrator.get_plugin(intent)
        
        if plugin and hasattr(plugin, "execute_async"):
            # Use async execution
            return await plugin.execute_async(parsed_data, context)
        else:
            # Fall back to synchronous execution
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.plugin_orchestrator.execute_plugin(intent, parsed_data, context)
            )
            
    def process_input(self, input_text: str, input_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Synchronous version of process_input for backward compatibility.
        
        Args:
            input_text: User input text
            input_context: Additional context information
            
        Returns:
            Response with generated text and metadata
        """
        # Create event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the async version and wait for result
        return loop.run_until_complete(self.process_input_async(input_text, input_context))
            
    async def shutdown(self):
        """Gracefully shut down the system integrator and its components."""
        logger.info("Shutting down system integrator...")
        self.persona_say("Initiating graceful shutdown of all subsystems, My Lord.")
        
        # Signal task workers to stop
        for _ in range(len(self.processing_tasks)):
            self.task_queue.put_nowait({"type": "shutdown"})
            
        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # Close resources
        if hasattr(self.memory, "close"):
            self.memory.close()
            
        if hasattr(self.knowledge_base, "close"):
            self.knowledge_base.close()
            
        logger.info("System integrator shutdown complete")
        self.persona_say("All subsystems have been shut down. At your leisure, My Lord.")

# Create singleton instance
_system_integrator = None

def get_system_integrator(config: Optional[Dict[str, Any]] = None):
    """Get singleton instance of SystemIntegrator."""
    global _system_integrator
    if _system_integrator is None:
        _system_integrator = SystemIntegrator(config)
    return _system_integrator