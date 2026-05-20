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
        user_query = self.get_user_query(context)
        user_prompt = self.user_message(context)
        print("SelectBestQueryStep system_prompt:\n", system_prompt)
        print("SelectBestQueryStep user_prompt:\n", user_prompt)
        messages = self.create_messages(system_prompt, [], [], user_prompt)

        plan = None
        for attempt in range(max_attempts):
            try:
                raw_response = await self.llm_client.chat_json(messages)
                print("SelectBestQueryStep raw_response:\n", raw_response)
                payload = json.loads(self._strip_fences(raw_response))
                plan = self.SelectionPlan.model_validate(payload)
                break
            except Exception as e:
                messages.append({"role": "assistant", "content": raw_response})
                error_message = f"Invalid JSON or schema: {str(e)}. Return ONLY JSON with 'reason' and 'selected_query'."
                messages.append({"role": "user", "content": error_message})

        if not plan:
            queries_to_analyze = self.get_search_results(context)
            if not queries_to_analyze:
                queries_to_analyze = [SearchResult(user_query, "", user_query)]
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

    def user_message(self, context: StepContext) -> str:
        user_query = self.get_user_query(context)
        validation_condition = self.get_validation_condition(context)
        user_intent = self.get_next_step_query(context)
        queries_to_analyze = self.get_search_results(context)
        if not queries_to_analyze:
            queries_to_analyze = [SearchResult(user_query, "", user_query)]
        queries_text = [r.text for r in queries_to_analyze]
        output_data = {
            "meta": {
                "agent": "search_query_validator"
            },
            "task": {
                "user_message": user_query,
                "required_criteria": validation_condition,
                "normalized_intent": user_intent,
                "queries_to_analyze": queries_text
            }
        }

        return json.dumps(output_data, ensure_ascii=False, indent=2)

    def stage(self) -> str:
        return "web_query_selection"

    def get_system_prompt(self, context: StepContext = None):
        return self.set_prompt_templates(
            "You are a search expert. Your job is to analyze the user's normalized intent, mandatory criteria, and a "
            "provided list of candidate search queries (\"queries_to_analyze\"), then select or adapt the "
            "single best query.\n\n"
            "Context: Current date is ${current_date}.\n\n"
            "INPUT STRUCTURE:\n"
            "You will receive a JSON containing:\n"
            "- \"task\": Includes \"user_message\", \"required_criteria\" (constraints), \"normalized_intent\" (global "
            "goal), and \"queries_to_analyze\" (list of candidate queries)."
            "RULES FOR QUERY SELECTION & ADAPTATION:\n"
            "1. Alignment: Select or adapt a query from \"queries_to_analyze\" that best represents the comprehensive "
            "concept matching the \"normalized_intent\".\n"
            "2. Strict Constraints: Stick ONLY to the parameters mentioned in \"required_criteria\" and "
            "\"normalized_intent\". Do NOT invent, assume, or add any extra criteria, filters, "
            "or sub-topics not explicitly requested.\n"
            "3. Language Match: The final \"selected_query\" MUST be written in the language of the \"user_message\". "
            "If the best candidate from the list is in a different language, you MUST translate its structure and "
            "keywords into the user's language.\n"
            "4. Technical Terms: Product names, brand names, or specific technical terms can remain in English if it "
            "ensures better search engine precision.\n"
            "5. Absolute Dates: Ensure any relative time terms are resolved into absolute values based on the current "
            "date (${current_date}) and match the \"required_criteria\".\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a JSON object. No markdown blocks, no extra text.\n"
            "{\n"
            "  \"reason\": \"Short explanation in English of why this query was chosen/adapted and if translation "
            "was applied\",\n"
            "  \"selected_query\": \"The final optimized and formatted query string in the user's language\""
            "}"
        )
