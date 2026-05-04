from abc import abstractmethod
import re

from handlers.pipelines.steps.pipeline_context import PipelineContext
from handlers.pipelines.steps.step_result import StepResult


class PipelineStep:

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
    def get_user_query(context: PipelineContext = None):
        if not context:
            raise ValueError("context is missing or empty")
        user_input = context.user_input
        if not user_input:
            raise ValueError("user_input is missing or empty")
        return user_input

    @staticmethod
    def get_search_results(context: PipelineContext = None):
        if not context:
            raise ValueError("context is missing or empty")
        search_results = context.search_results
        return search_results

    @staticmethod
    def get_validation_condition(context: PipelineContext = None):
        if not context:
            raise ValueError("context is missing or empty")
        validation_condition = context.validation_condition
        if not validation_condition:
            raise ValueError("validation_condition is missing or empty")
        return validation_condition

    @staticmethod
    def get_intent(context: PipelineContext = None):
        if not context:
            raise ValueError("context is missing or empty")
        intent = context.intent
        if not intent:
            raise ValueError("intent is missing or empty")
        return intent

    @staticmethod
    def get_answer(context: PipelineContext = None):
        if not context:
            raise ValueError("context is missing or empty")
        answer = context.answer
        if not answer:
            raise ValueError("answer is missing or empty")
        return answer

    @staticmethod
    def get_sources(context: PipelineContext = None):
        if not context:
            raise ValueError("context is missing or empty")
        sources = context.sources
        return sources

    @staticmethod
    def get_validation_issues(context: PipelineContext = None):
        if not context:
            raise ValueError("context is missing or empty")
        validation_issues = context.validation_issues
        if not validation_issues:
            raise ValueError("validation_issues is missing or empty")
        return validation_issues

    @staticmethod
    def get_suggestions(context: PipelineContext = None):
        if not context:
            raise ValueError("context is missing or empty")
        suggestions = context.suggestions
        return suggestions

    @staticmethod
    def get_search_queries(context: PipelineContext = None):
        if not context:
            raise ValueError("context is missing or empty")
        search_queries = context.search_queries
        if not search_queries:
            search_queries = []
        return search_queries

    @staticmethod
    def get_system_prompt(context: PipelineContext = None):
        if not context:
            raise ValueError("context is missing or empty")
        return context.system_prompt

    @staticmethod
    def get_history(context: PipelineContext = None):
        history = []
        if context and context.history:
            return context.history
        return history

    def system_prompt(self):
        return None

    def get_suggestions_message(self, context):
        user_query = self.get_user_query(context)
        suggestions = self.get_suggestions(context)
        if suggestions:
            feedback_msg = "Your previous search plan failed validation.\n"
            if user_query:
                feedback_msg += f"- Used queries: {user_query}\n"
            answer = self.get_answer(context)
            if answer:
                feedback_msg += f"- Generated answer was insufficient: {answer}\n"
            issues = self.get_validation_issues(context)
            if issues:
                feedback_msg += f"- Issues found: {"\n".join(issues)}\n"
            feedback_msg += f"- Suggestions for improvement: {suggestions}\n"

            feedback_msg += "\nPlease generate a NEW search plan taking this feedback into account. Adjust search queries or strategy."
            return {"role": "user", "content": feedback_msg}


    def create_messages(self, context) -> list[dict]:
        messages = []

        system_prompt = self.system_prompt()
        if not system_prompt:
            system_prompt = self.get_system_prompt(context)
        messages.append({"role": "system", "content": system_prompt})

        history = self.get_history(context)
        messages.extend(history)

        user_query = self.get_user_query(context)
        suggestions_msg = self.get_suggestions_message(context)
        if suggestions_msg:
            messages.append(suggestions_msg)

        messages.append({"role": "user", "content": user_query})

        return messages

    @staticmethod
    def _strip_fences(text: str) -> str:
        return re.sub(r"```(?:json)?\n?|```", "", text or "").strip()
