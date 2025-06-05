# core/command_handler.py

import logging
from core.intent_parser import parse_intent
from core.task_dispatcher import TaskDispatcher

class CommandHandler:
    def __init__(self, task_dispatcher: TaskDispatcher):
        self.task_dispatcher = task_dispatcher
        self.logger = logging.getLogger(__name__)

    def handle_command(self, user_input: str, context: dict) -> str:
        """
        Handle user command by parsing intent and dispatching the appropriate task.

        Args:
            user_input (str): Raw user input string.
            context (dict): Contextual info about current conversation/session.

        Returns:
            str: Response text from executed task or error message.
        """
        try:
            intent = parse_intent(user_input, context)
            if not intent:
                return "I am afraid I could not understand your request."

            # Dispatch intent to appropriate plugin/task handler
            response = self.task_dispatcher.dispatch(intent, context)
            return response or "No response was generated from the command."
        except Exception as e:
            self.logger.error(f"Error handling command: {e}", exc_info=True)
            return "An unexpected error occurred while processing your command."

