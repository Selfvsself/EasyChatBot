import json

from pydantic import BaseModel, Field

from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.step_context import StepContext
from handlers.pipelines.steps.step_result import StepResult


class SearchQueryStep(BaseStep):
    class SearchPlan(BaseModel):
        search_query: str = Field(description="Optimized search engine query")
        reason: str = Field(description="Brief explanation of why this query was chosen")

    async def parse_decision_with_retry(self, context: StepContext) -> SearchPlan:
        max_attempts = 3
        system_prompt = self.get_system_prompt(context)
        history = self.get_history(context)
        user_query = self.get_user_query(context)
        required_info = self.get_validation_condition(context)
        internal_messages = self.get_internal_messages(context)
        prompt_with_history = self.user_message(user_query, required_info, internal_messages, history)
        messages = self.create_messages(system_prompt, [], [], prompt_with_history)

        plan = None
        for attempt in range(max_attempts):
            try:
                raw_response = await self.llm_client.chat_json(messages)
                payload = json.loads(self._strip_fences(raw_response))
                plan = self.SearchPlan.model_validate(payload)
                break
            except Exception as e:
                messages.append({"role": "assistant", "content": raw_response})
                error_message = f"Invalid JSON or schema: {str(e)}. Return ONLY JSON with 'reason' and 'search_query'."
                messages.append({"role": "user", "content": error_message})

        if not plan:
            plan = self.SearchPlan(
                search_query=user_query,
                reason="fallback to original user query due to parsing error"
            )

        return plan

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")

        plan = await self.parse_decision_with_retry(context)

        result_ctx = StepContext.from_context(context)
        result_ctx.next_step_query = plan.search_query

        # internal_messages = self.get_internal_messages(context)
        # internal_messages.append(f"SEARCH QUERY GENERATED: {plan.search_query}")
        # result_ctx.internal_messages = internal_messages

        return StepResult(context=result_ctx, stop=False)

    @staticmethod
    def user_message(user_query: str, required_info: str, internal_messages: list[str], history: list[dict]):
        internal_message = "None"
        if internal_messages:
            internal_message = "<thought>\n"
            for idx, msg in enumerate(internal_messages):
                internal_message += f"Step {idx + 1}:\n"
                internal_message += msg
                internal_message += "\n"
            internal_message += "</thought>"

        history_text = "\n".join(f"- Role: '{m["role"]}' Content: '{m["content"]}'" for m in history)

        return (
            f"Conversation history: \n<history>\n{history_text}\n</history>\n\n"
            f"Original user intent: {user_query}\n"
            f"Required Criteria: {required_info}\n\n"
            f"History of thoughts: {internal_message}\n\n"
        )

    def stage(self) -> str:
        return "web_query_planning"

    def get_system_prompt(self, context: StepContext = None):
        return self.set_prompt_templates(
            "You are a search expert. Your job is to analyze the user query and create ONE optimized "
            "search engine query to find the most relevant information.\n\n"
            "Context: Current date is ${current_date}.\n\n"
            "RULES:\n"
            "- Transform the user's intent into a concise, keyword-rich search query.\n"
            "- Determine the language the user speaks and use that same language for the 'search_query'.\n"
            "- If the query is technical, you may include relevant English terms for better results.\n\n"
            "- Try to formulate your query in the user's language so that the search returns the most relevant results.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            "  \"reason\": \"short explanation in English\",\n"
            "  \"search_query\": \"the optimized query string\"\n"
            "}"
        )
