import json
from enum import StrEnum
from pydantic import BaseModel, Field
from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.step_context import StepContext, SearchResult
from handlers.pipelines.steps.step_result import StepResult


class ValidationAction(StrEnum):
    SEARCH = "SEARCH"  # Придумать другой запрос
    CLARIFY = "CLARIFY"  # Поискать ещё результаты с этим же запросом
    RESPOND = "RESPOND"  # Перегенерировать ответ с текущими данными


class SearchValidationStep(BaseStep):
    class ValidationResponse(BaseModel):
        passed: bool
        issues: str
        action: ValidationAction = Field(..., description="Action to take based on validation")
        action_payload: str = Field("", description="Payload for action (e.g., new search query)")

    async def validate_response_with_retry(self, context: StepContext) -> ValidationResponse:
        system_prompt = self.get_system_prompt(context)
        history = self.get_history(context)
        user_query = self.get_user_query(context)
        current_answer = self.get_next_step_query(context)
        validation_condition = self.get_validation_condition(context)
        search_results = self.get_search_results(context)
        user_message = self.user_message(user_query, validation_condition, current_answer, search_results)
        internal_messages = self.get_internal_messages(context)
        messages = self.create_messages(system_prompt, history, internal_messages, user_message)
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                raw_response = await self.llm_client.chat_json(messages)
                messages.append({"role": "assistant", "content": raw_response})
                payload = json.loads(self._strip_fences(raw_response))
                return self.ValidationResponse.model_validate(payload)
            except Exception as e:
                error_message = (
                    f"Your previous response caused a parsing error: {str(e)}. "
                    f"Please correct the output and return ONLY valid JSON matching the required schema."
                )
                messages.append({"role": "user", "content": error_message})

        return self.ValidationResponse(
            passed=False,
            issues="Failed to parse validator response after max attempts",
            action=ValidationAction.RESPOND,
            action_payload="Fallback: system failed to parse response."
        )

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")

        search_results = self.get_search_results(context)
        if not search_results:
            return StepResult(context=context, stop=False)

        validation_result = await self.validate_response_with_retry(context)

        result_ctx = StepContext.from_context(context)

        internal_messages = self.get_internal_messages(context)
        internal_messages.append(
            "VALIDATION AGENT\n"
            f"VALIDATION RESULT:\n{validation_result.passed}\n"
            f"FOUND ISSUES: {validation_result.issues}\n"
            f"ACTION: {validation_result.action}\n"
            f"ACTION PAYLOAD: {validation_result.action_payload}\n"
        )
        result_ctx.internal_messages = internal_messages

        result_ctx.action = validation_result.action

        is_stop = False
        is_success = False

        if validation_result.passed:
            is_stop = True
            is_success = True

        return StepResult(context=result_ctx, stop=is_stop, success=is_success)

    @staticmethod
    def user_message(user_query: str, stop_condition: str, current_answer: str, sources: list[SearchResult]):
        sources_text = ""
        for site in sources:
            sources_text += f" - Title: {site.title}, URL: {site.url}\n"
        return (
            f"User Original Query: {user_query}\n"
            f"Stop/Success Condition: {stop_condition}\n\n"
            f"Generated Answer to Validate:\n{current_answer}\n"
            f"Sources:\n{sources_text}"
        )

    def stage(self) -> str:
        return "web_validating"

    def get_system_prompt(self, context: StepContext = None):
        return (
            "You are a response validator.\n"
            "Your job is to check if the generated answer satisfies the search plan and stop conditions.\n\n"

            "RULES:\n"
            "- Check if all required info from the plan is present in the answer\n"
            "- Check if the stop/success condition is fully met\n"
            "- If the answer is incomplete or incorrect, set 'passed' to false and choose the appropriate 'action'\n"
            "- Do NOT answer the question yourself\n"
            "- Only return JSON\n\n"

            "ACTIONS EXPLANATION (Choose one):\n"
            "- 'SEARCH': The current search results are insufficient or irrelevant. Suggest a completely NEW search query in 'action_payload'.\n"
            "- 'CLARIFY': The current search query is good, but you need to fetch more/deeper results for the SAME query. Put the current query in 'action_payload'.\n"
            "- 'RESPOND': The available data is enough, but the assistant made a mistake generating the answer. Describe what to fix in 'issues' and leave 'action_payload' empty.\n\n"

            "OUTPUT JSON SCHEMA:\n"
            "{\n"
            "  \"passed\": bool,\n"
            "  \"issues\": \"Specific details on what is missing or incorrect, or empty if passed is true.\",\n"
            "  \"action\": \"SEARCH\" | \"CLARIFY\" | \"RESPOND\",\n"
            "  \"action_payload\": \"Your new query or current query or additional instruction\"\n"
            "}"
        )
