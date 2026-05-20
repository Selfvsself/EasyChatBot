import json

from pydantic import BaseModel

from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.step_context import StepContext
from handlers.pipelines.steps.step_result import StepResult


class GenerateStep(BaseStep):
    class WebAnswer(BaseModel):
        final_response: str

    def parse_summary_result(self, raw):
        try:
            payload = json.loads(self._strip_fences(raw))
            return self.WebAnswer.model_validate(payload)
        except Exception:
            return self.WebAnswer(
                final_response="Something went wrong, try again later"
            )

    async def get_answer_with_retry(self, context: StepContext) -> WebAnswer | None:
        summary = None
        max_attempts = 3
        system_prompt = self.get_system_prompt(context)
        user_prompt = self.user_message(context)
        print("GenerateAnswerStep system_prompt:\n", system_prompt)
        print("GenerateAnswerStep user_prompt:\n", user_prompt)
        messages = self.create_messages(system_prompt, [], user_prompt)
        for attempt in range(max_attempts):
            try:
                raw_response = await self.llm_client.chat_json(messages)
                print("GenerateAnswerStep raw_response:\n", raw_response)
                messages.append({"role": "assistant", "content": raw_response})
                payload = json.loads(self._strip_fences(raw_response))
                summary = self.WebAnswer.model_validate(payload)
                break
            except Exception as e:
                error_message = (
                    f"Your previous response caused a parsing error: {str(e)}. "
                    f"Please correct the output and return ONLY valid JSON matching the required schema."
                )
                messages.append({"role": "user", "content": error_message})

        if not summary:
            summary = self.WebAnswer(
                final_response="Something went wrong, try again later"
            )
        return summary

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")

        answer = await self.get_answer_with_retry(context)
        result_ctx = StepContext.from_context(context)
        result_ctx.answer = answer.final_response

        return StepResult(context=result_ctx, stop=False)

    def user_message(self, context: StepContext) -> str:
        user_query = self.get_user_query(context)
        validation_condition = self.get_validation_condition(context)
        history = self.get_history(context)
        chat_memory = self.get_chat_memory(context)

        user_intent = self.get_user_intent(context)
        output_data = {
            "meta": {
                "agent": "answer_agent"
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
        return "typing"

    def stage_metadata(self) -> dict:
        return {}
