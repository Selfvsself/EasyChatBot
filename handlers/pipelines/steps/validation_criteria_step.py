import json
import logging
from datetime import datetime

from pydantic import BaseModel

from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.step_context import StepContext
from handlers.pipelines.steps.step_result import StepResult


class ValidationCriteriaStep(BaseStep):
    class CriteriaResponse(BaseModel):
        completeness_criteria: str

    async def _get_criteria_with_retry(self, context: StepContext) -> str:
        max_attempts = 3
        system_prompt = self.get_system_prompt(context)
        history = self.get_history(context)
        user_query = self.get_user_query(context)
        internal_messages = self.get_internal_messages(context)
        messages = self.create_messages(system_prompt, history, internal_messages, user_query)

        for attempt in range(max_attempts):
            try:
                raw_response = await self.llm_client.chat_json(messages)
                payload = json.loads(self._strip_fences(raw_response))
                validated_data = self.CriteriaResponse.model_validate(payload)
                return validated_data.completeness_criteria

            except Exception as e:
                logging.warning(f"ValidationCriteriaStep attempt {attempt + 1} failed: {e}")
                error_message = (
                    f"Your previous response caused a parsing error: {str(e)}. "
                    "Return ONLY valid JSON with 'completeness_criteria' key."
                )
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({"role": "user", "content": error_message})

        return "The request is fulfilled according to the user's initial intent."

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        if not context:
            return StepResult(context=context, stop=False)

        criteria = await self._get_criteria_with_retry(context)
        result_ctx = StepContext.from_context(context)

        result_ctx.validation_condition = criteria
        internal_messages = self.get_internal_messages(context)
        internal_messages.append(f"VALIDATION CRITERIA: {criteria}")

        result_ctx.internal_messages = internal_messages

        return StepResult(context=result_ctx, stop=False)

    def stage(self) -> str:
        return "validation_planning"

    def get_system_prompt(self, context: StepContext = None):
        current_date = datetime.now().strftime("%A, %d %B %Y")
        action = self.get_action(context)
        next_step = self.get_next_step_query(context)
        if not next_step:
            next_step = self.get_user_query(context)
        return (
            "You are a Quality Assurance Specialist.\n"
            "Your task is to define a single, clear criterion for checking if a task is completed.\n\n"
            f"PLANNED ACTION: {action}\n"
            f"TARGET STEP: {next_step}\n\n"
            f"Context: Current date is {current_date}.\n\n"
            "RULES:\n"
            "1. Language: Use the same language as the TARGET STEP.\n"
            "2. Precision: The criteria must be specific to the facts or actions mentioned.\n"
            "3. Format: Return ONLY JSON: {\"completeness_criteria\": \"string\"}.\n"
            "4. Content: Do not add new tasks. Only describe how to verify the current TARGET STEP."
        )
