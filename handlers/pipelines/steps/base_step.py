import re
from abc import abstractmethod
from datetime import datetime

from handlers.pipelines.steps.enum.step_action import StepAction
from handlers.pipelines.steps.step_context import StepContext
from handlers.pipelines.steps.step_result import StepResult


class BaseStep:

    def __init__(self, llm_client):
        self.llm_client = llm_client

    @abstractmethod
    async def execute(self, context, stage_callback=None) -> StepResult:
        pass

    def stage(self) -> str:
        return "thinking"

    def stage_metadata(self) -> dict:
        return {}

    @staticmethod
    async def _emit_stage(stage_callback, stage: str, metadata: dict | None = None):
        if stage_callback:
            await stage_callback(stage=stage, metadata=metadata or {})

    @staticmethod
    def _strip_fences(text: str) -> str:
        return re.sub(r"```(?:json)?\n?|```", "", text or "").strip()

    @staticmethod
    def get_user_query(context: StepContext = None):
        if not context:
            raise ValueError("context is missing or empty")
        user_input = context.user_input
        if not user_input:
            raise ValueError("user_input is missing or empty")
        return user_input

    def get_system_prompt(self, context: StepContext = None):
        if not context:
            raise ValueError("context is missing or empty")
        return context.system_prompt

    @staticmethod
    def get_validation_condition(context: StepContext = None):
        validation_condition = "None"
        if context and context.validation_condition:
            validation_condition = context.validation_condition
        return validation_condition

    @staticmethod
    def get_history(context: StepContext = None):
        history = []
        if context and context.history:
            return context.history
        return history

    @staticmethod
    def get_chat_memory(context: StepContext = None):
        chat_memory = []
        if context and context.chat_memory:
            return context.chat_memory
        return chat_memory

    @staticmethod
    def get_next_step_query(context: StepContext = None):
        if not context:
            raise ValueError("context is missing or empty")
        next_step_query = "None"
        if context.next_step_query:
            next_step_query = context.next_step_query
        return next_step_query

    @staticmethod
    def get_search_results(context: StepContext = None):
        search_results = []
        if context and context.search_results:
            return context.search_results
        return search_results

    @staticmethod
    def get_action(context: StepContext = None):
        action = StepAction.RESPOND
        if context:
            return context.action
        return action

    @staticmethod
    def get_internal_messages(context: StepContext = None):
        internal_messages = []
        if context and context.internal_messages:
            return context.internal_messages
        return internal_messages

    def create_messages(self,
                        system_prompt: str,
                        history: list[dict],
                        internal_messages: list[str],
                        user_query: str) -> list[dict]:
        messages = []

        if not system_prompt:
            raise ValueError("system_prompt is missing or empty")
        messages.append({"role": "system", "content": system_prompt})

        if history:
            messages.extend(history)

        if internal_messages:
            internal_message = "<thought>\n"
            for idx, msg in enumerate(internal_messages):
                internal_message += f"Step {idx + 1}:\n"
                internal_message += msg
                internal_message += "\n"
            internal_message += "</thought>"

            messages.append({"role": "assistant", "content": internal_message})

        if user_query:
            messages.append({"role": "user", "content": user_query})

        return messages

    @staticmethod
    def set_prompt_templates(prompt: str) -> str:
        CURRENT_DATE_TEMPLATE = "${current_date}"
        current_date_value = datetime.now().strftime("%A, %d %B %Y")
        text = prompt
        if CURRENT_DATE_TEMPLATE in prompt:
            text = prompt.replace(CURRENT_DATE_TEMPLATE, current_date_value)
        return text
