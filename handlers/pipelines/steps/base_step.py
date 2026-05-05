import re
from abc import abstractmethod

from handlers.pipelines.steps.step_result import StepResult
from handlers.pipelines.steps.step_context import StepContext


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

    def system_prompt(self):
        return None

    @staticmethod
    def get_system_prompt(context: StepContext = None):
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
    def get_next_step_query(context: StepContext = None):
        if not context:
            raise ValueError("context is missing or empty")
        next_step_query = context.next_step_query
        if not next_step_query:
            raise ValueError("next_step_query is missing or empty")
        return next_step_query

    @staticmethod
    def get_search_results(context: StepContext = None):
        search_results = []
        if context and context.search_results:
            return context.search_results
        return search_results

    @staticmethod
    def get_internal_messages(context: StepContext = None):
        internal_messages = []
        if context and context.internal_messages:
            return context.internal_messages
        return internal_messages

    def create_messages(self,
                        system_prompt: str,
                        history: list[dict],
                        internal_messages:list[str],
                        user_query: str) -> list[dict]:
        messages = []

        if not system_prompt:
            raise ValueError("system_prompt is missing or empty")
        messages.append({"role": "system", "content": system_prompt})

        if history:
            messages.extend(history)

        if user_query:
            messages.append({"role": "user", "content": user_query})

        if internal_messages:
            internal_message = "<thought>\n"
            for idx, msg in enumerate(internal_messages):
                internal_message += f"Step {idx + 1}:\n"
                internal_message += msg
                internal_message += "\n"
            internal_message += "</thought>"

            messages.append({"role": "assistant", "content": internal_message})

        return messages
