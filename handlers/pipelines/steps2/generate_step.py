from handlers.pipelines.steps2.step_context import StepContext
from handlers.pipelines.steps2.base_step import BaseStep
from handlers.pipelines.steps2.step_result import StepResult


class GenerateStep(BaseStep):
    async def get_answer_with_retry(self, messages: list[dict], user_query: str) -> str | None:
        answer = None
        max_attempts = 5
        for attempt in range(max_attempts):
            raw_response = await self.llm_client.chat_text(messages)
            if raw_response:
                answer = raw_response
                break
            error_message = (
                f"The last message from you was empty, perhaps you were trying to call the tool"
                f"Please fix the output and return ONLY the text. Don't call the any tools."
            )
            messages.append({"role": "user", "content": error_message})

        if not answer:
            answer = user_query
        return answer

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")
        system_prompt = self.get_system_prompt(context)
        history = self.get_history(context)
        user_query = self.get_user_query(context)
        internal_messages = self.get_internal_messages(context)
        messages = self.create_messages(system_prompt, history, internal_messages, user_query)
        messages.append({"role": "user", "content": user_query})
        answer = await self.get_answer_with_retry(messages, user_query)
        result_ctx = StepContext.from_context(context)
        result_ctx.answer = answer

        internal_messages = self.get_internal_messages(context)
        internal_messages.append(
            "RESPONSE AGENT\n"
            f"ANSWER:\n{answer}\n"
        )
        result_ctx.internal_messages = internal_messages
        return StepResult(context=result_ctx, stop=False)

    def stage(self) -> str:
        return "typing"

    def stage_metadata(self) -> dict:
        return {}
