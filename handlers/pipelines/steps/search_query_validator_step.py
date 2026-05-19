import json
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.step_context import StepContext, SearchResult
from handlers.pipelines.steps.step_result import StepResult


class SelectBestQueryStep(BaseStep):
    class SelectionPlan(BaseModel):
        selected_query: str = Field(description="The most suitable, comprehensive query from the provided list")
        reason: str = Field(description="Brief explanation of why this query was selected, written in English")


    async def parse_decision_with_retry(self, context: StepContext) -> SelectionPlan:
        max_attempts = 3
        system_prompt = self.get_system_prompt(context)
        history = self.get_history(context)
        user_query = self.get_user_query(context)
        internal_messages = self.get_internal_messages(context)
        required_info = self.get_validation_condition(context)

        queries_to_analyze = self.get_search_results(context)

        if not queries_to_analyze:
            queries_to_analyze = [SearchResult(user_query, "", user_query)]

        prompt_with_candidates = self.user_message(user_query, required_info, internal_messages, history, queries_to_analyze)

        messages = self.create_messages(system_prompt, [], [], prompt_with_candidates)

        plan = None
        for attempt in range(max_attempts):
            try:
                raw_response = await self.llm_client.chat_json(messages)
                payload = json.loads(self._strip_fences(raw_response))
                plan = self.SelectionPlan.model_validate(payload)
                break
            except Exception as e:
                messages.append({"role": "assistant", "content": raw_response})
                error_message = f"Invalid JSON or schema: {str(e)}. Return ONLY JSON with 'reason' and 'selected_query'."
                messages.append({"role": "user", "content": error_message})

        if not plan:
            plan = self.SelectionPlan(
                selected_query=queries_to_analyze[0].text,
                reason="fallback to first candidate due to parsing or validation error"
            )

        return plan

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")

        plan = await self.parse_decision_with_retry(context)

        result_ctx = StepContext.from_context(context)
        result_ctx.next_step_query = plan.selected_query

        internal_messages = self.get_internal_messages(context)
        internal_messages.append(f"WEB QUERY SELECTED: {plan.selected_query}")
        result_ctx.internal_messages = internal_messages

        return StepResult(context=result_ctx, stop=False)

    @staticmethod
    def user_message(user_query: str, required_info: str, internal_messages: list[str], history: list[dict], queries_to_analyze: list[SearchResult]):
        internal_message = "None"
        if internal_messages:
            internal_message = "<thought>\n"
            for idx, msg in enumerate(internal_messages):
                internal_message += f"Step {idx + 1}:\n"
                internal_message += msg
                internal_message += "\n"
            internal_message += "</thought>"

        history_text = "\n".join(f"- Role: '{m["role"]}' Content: '{m["content"]}'" for m in history)
        queries_list_str = "\n".join(f"- {q.text}" for q in queries_to_analyze)

        return (
            f"Conversation history: \n<history>\n{history_text}\n</history>\n\n"
            f"Original user intent: {user_query}\n"
            f"Required Criteria: {required_info}\n\n"
            f"History of thoughts: {internal_message}\n\n"
            f"Candidate queries:\n{queries_list_str}"
        )

    def stage(self) -> str:
        return "web_query_selection"

    def get_system_prompt(self, context: StepContext = None):
        return self.set_prompt_templates(
            "You are a search expert. Your job is to analyze the user's original intent and the provided list of "
            "candidate search queries, then select or adapt the single best query.\n\n"
            "Context: Current date is ${current_date}.\n\n"
            "RULES:\n"
            "- Select the most suitable query that represents the most general/comprehensive concept matching the user's intent.\n"
            "- Strictly stick ONLY to the parameters mentioned by the user. Do NOT invent, assume, or add any extra criteria, constraints, or sub-topics that the user did not explicitly ask for.\n"
            "- Determine the language the user is speaking in 'Original user intent'.\n"
            "- The final 'selected_query' MUST be in the user's language. If the best query from the candidates list is written in a different language, you MUST translate it into the user's language.\n"
            "- Technical terms or product names can remain in English if it helps search precision, but the query structure must match the user's language.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            "  \"reason\": \"short explanation in English why this query was chosen and if translation was applied\",\n"
            "  \"selected_query\": \"the final optimized query string in the user's language\"\n"
            "}"
        )
