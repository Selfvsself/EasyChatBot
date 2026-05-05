import json
from datetime import datetime

from pydantic import BaseModel, Field

from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.step_context import StepContext
from handlers.pipelines.steps.step_result import StepResult


class SearchQueryStep(BaseStep):
    class SearchPlan(BaseModel):
        search_query: str = Field(description="Optimized search engine query")
        reason: str = Field(description="Brief explanation of why this query was chosen")

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        system_prompt = self.get_system_prompt(context)
        history = self.get_history(context)
        user_query = self.get_user_query(context)
        internal_messages = self.get_internal_messages(context)
        messages = self.create_messages(system_prompt, history, internal_messages, user_query)

        try:
            raw_response = await self.llm_client.chat_json(messages)
            payload = json.loads(self._strip_fences(raw_response))
            plan = self.SearchPlan.model_validate(payload)
        except Exception:
            plan = self.SearchPlan(
                search_query=user_query,
                reason="fallback to original user query due to parsing error"
            )

        result_ctx = StepContext.from_context(context)
        result_ctx.next_step_query = plan.search_query

        internal_messages.append(f"SEARCH QUERY GENERATED: {plan.search_query}")
        result_ctx.internal_messages = internal_messages

        return StepResult(context=result_ctx, stop=False)

    def stage(self) -> str:
        return "web_query_planning"

    def get_system_prompt(self, context: StepContext = None):
        current_date = datetime.now().strftime("%A, %d %B %Y")
        return (
            "You are a search expert. Your job is to analyze the user query and create ONE optimized "
            "search engine query to find the most relevant information.\n\n"
            f"Context: Current date is {current_date}.\n\n"
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
