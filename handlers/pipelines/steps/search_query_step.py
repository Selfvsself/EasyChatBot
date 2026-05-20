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
        user_query = self.get_user_query(context)
        user_prompt = self.user_message(context)
        print("SearchQueryStep system_prompt:\n", system_prompt)
        print("SearchQueryStep user_prompt:\n", user_prompt)
        messages = self.create_messages(system_prompt, [], [], user_prompt)

        plan = None
        for attempt in range(max_attempts):
            try:
                raw_response = await self.llm_client.chat_json(messages)
                print("SearchQueryStep raw_response:\n", raw_response)
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

        return StepResult(context=result_ctx, stop=False)

    def user_message(self, context: StepContext) -> str:
        history = self.get_history(context)
        user_query = self.get_user_query(context)
        validation_condition = self.get_validation_condition(context)
        user_intent = self.get_next_step_query(context)
        chat_memory = self.get_chat_memory(context)
        output_data = {
            "meta": {
                "agent": "search_planner"
            },
            "task": {
                "user_message": user_query,
                "required_criteria": validation_condition,
                "normalized_intent": user_intent
            },
            "context": {
                "recent_history": history,
                "chat_memory": chat_memory
            }
        }

        return json.dumps(output_data, ensure_ascii=False, indent=2)

    def stage(self) -> str:
        return "web_query_planning"

    def get_system_prompt(self, context: StepContext = None):
        return self.set_prompt_templates(
            "You are a search expert. Your job is to analyze the user intent and constraints, then create ONE optimized"
            " search engine query to find the most relevant and accurate information.\n\n"
            "Context: Current date is ${current_date}.\n\n"
            "INPUT STRUCTURE:\n"
            "You will receive a JSON containing:\n"
            "- \"task\": Includes \"user_message\" (current text), \"required_criteria\" (constraints), and "
            "\"normalized_intent\" (reconstructed global goal).\n"
            "- \"context\": \"recent_history\" and \"chat_memory\".\n\n"
            "RULES FOR SEARCH QUERY GENERATION:\n"
            "1. Base on Intent & Criteria: Transform the \"normalized_intent\" and \"required_criteria\" into a concise"
            ", keyword-rich search query. Ignore conversational fluff.\n"
            "2. Language: Formulate the \"search_query\" in the language of the \"user_message\" to get the most "
            "relevant local results. If the topic is highly technical, medical, or global, you may mix or use English "
            "terms to improve results.\n"
            "3. Temporal Relevance: Use the current date (${current_date}) to accurately resolve relative time terms "
            "(like \"next month\", \"latest\", \"this year\") into absolute numbers/years inside the \"search_query\"."
            "4. Format: Do not use advanced search operators (like site:, filetype:, OR) unless absolutely necessary. "
            "Focus on high-quality keywords.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a JSON object. No markdown blocks, no extra text.\n"
            "{\n"
            "\"reason\": \"Short explanation of the search query logic in English\",\n"
            "\"search_query\": \"The optimized search engine query string\"\n"
            "}"
        )
