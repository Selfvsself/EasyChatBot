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
        user_prompt = self.user_message(context)
        messages = self.create_messages(system_prompt, [], user_prompt)

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

        return StepResult(context=result_ctx, stop=False)

    def user_message(self, context: StepContext) -> str:
        history = self.get_history(context)
        user_query = self.get_user_query(context)
        validation_condition = self.get_validation_condition(context)
        user_intent = self.get_user_intent(context)
        chat_memory = self.get_chat_memory(context)
        output_data = {
            "meta": {
                "agent": "validation_criteria_planner"
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
        return "validation_planning"

    def get_system_prompt(self, context: StepContext = None):
        return self.set_prompt_templates(
            "You are a Quality Assurance Specialist. Your task is to define a single, clear, and objective criterion "
            "for checking if the current task or step is successfully completed.\n\n"
            "Context: Current date is ${current_date}.\n\n"
            "INPUT STRUCTURE:\n"
            "You will receive a JSON containing:\n"
            "- \"task\": Includes \"user_message\" (current text), \"required_criteria\" (constraints), and "
            "\"normalized_intent\" (reconstructed global goal).\n"
            "- \"context\": \"recent_history\" and \"chat_memory\".\n\n"
            "RULES FOR CRITERIA GENERATION:\n"
            "1. Base on Intent: Analyze \"normalized_intent\" and \"required_criteria\" to understand exactly what data "
            "or result must be delivered to the user.\n"
            "2. Precision & Factuality: The criteria must be specific, measurable, and directly tied to the facts, "
            "parameters, or actions mentioned in the query and constraints.\n"
            "3. No New Tasks: Only describe how to verify the fulfillment of the current intent. "
            "Do not add next steps or separate sub-tasks.\n"
            "4. Language: Write the \"completeness_criteria\" in the same language as the \"user_message\"."
            "5. Sources: DO NOT check for the presence of sources, links, or citations. Verification of references "
            "is completely ignored.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a JSON object. No markdown blocks, no extra text.\n"
            "{\n"
            "  \"completeness_criteria\": \"Single, precise sentence describing how to verify that the intent and "
            "constraints are fully met\"\n"
            "}"
        )
