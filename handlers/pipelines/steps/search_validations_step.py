import json
from pydantic import BaseModel
from handlers.pipelines.steps.pipeline_context import PipelineContext
from handlers.pipelines.steps.pipeline_step import PipelineStep
from handlers.pipelines.steps.step_result import StepResult


class SearchValidationStep(PipelineStep):
    class ValidationResponse(BaseModel):
        passed: bool
        issues: list[str]
        improvement_suggestions: str

    async def execute(self, context: PipelineContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")

        if not context.search_queries:
            return StepResult(context=context, stop=False)

        validation_result = await self.validate_response_with_retry(context)

        result_ctx = PipelineContext.from_context(context)

        if not validation_result.passed:
            result_ctx.validation_issues = validation_result.issues
            result_ctx.suggestions = validation_result.improvement_suggestions
            return StepResult(context=result_ctx, stop=True)

        return StepResult(context=result_ctx, stop=False)

    async def validate_response_with_retry(self, context: PipelineContext) -> ValidationResponse:
        messages = self.create_validation_messages(context)
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
            issues=["Failed to parse validator response after max attempts"],
            improvement_suggestions="Please retry the generation focusing on strict JSON schema."
        )

    def create_validation_messages(self, context: PipelineContext) -> list[dict]:
        user_query = self.get_user_query(context)

        intent = self.get_intent(context)
        stop_condition = self.get_validation_condition(context)
        current_answer = self.get_answer(context)
        sources = self.get_sources()
        if sources:
            current_answer += "\n"
            current_answer += "\n".join(sources)

        prompt = (
            f"User Original Query: {user_query}\n"
            f"Search Intent: {intent}\n"
            f"Stop/Success Condition: {stop_condition}\n\n"
            f"Generated Answer to Validate:\n{current_answer}"
        )

        return [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": prompt}
        ]

    def stage(self) -> str:
        return "web_validating"

    def system_prompt(self) -> str:
        return (
            "You are a response validator.\n"
            "Your job is to check if the generated answer satisfies the search plan and stop conditions.\n\n"

            "RULES:\n"
            "- Check if all required info from the plan is present in the answer\n"
            "- Check if the stop/success condition is fully met\n"
            "- If the answer is incomplete or incorrect, set 'passed' to false and list specific issues\n"
            "- Do NOT answer the question yourself\n"
            "- Only return JSON\n\n"

            "OUTPUT JSON SCHEMA:\n"
            "{\n"
            "  \"passed\": bool,\n"
            "  \"issues\": list[str],\n"
            "  \"improvement_suggestions\": str\n"
            "}"
        )
