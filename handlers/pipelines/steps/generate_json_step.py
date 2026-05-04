from handlers.pipelines.steps.pipeline_context import PipelineContext
from handlers.pipelines.steps.pipeline_step import PipelineStep
from handlers.pipelines.steps.step_result import StepResult


class GenerateJsonStep(PipelineStep):
    async def get_answer_with_retry(self, messages: list[dict], user_query: str) -> str | None:
        answer = None
        max_attempts = 5
        for attempt in range(max_attempts):
            raw_response = await self.llm_client.chat_json(messages)
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

    async def execute(self, context: PipelineContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")
        user_query = self.get_user_query(context)
        messages = self.create_messages(context)
        answer = await self.get_answer_with_retry(messages, user_query)
        result_ctx = PipelineContext.from_context(context)
        result_ctx.answer = answer
        return StepResult(context=result_ctx, stop=False)

    def stage(self) -> str:
        return "typing"

    def stage_metadata(self) -> dict:
        return {}
