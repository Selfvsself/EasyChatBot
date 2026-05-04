from handlers.pipelines.steps.pipeline_context import PipelineContext
from handlers.pipelines.steps.pipeline_step import PipelineStep
from handlers.pipelines.steps.step_result import StepResult
import logging


class SearchResponseStep(PipelineStep):

    async def get_answer_with_retry(self, messages: list[dict], combined_summaries: str) -> str | None:
        answer = None
        max_attempts = 5
        for attempt in range(max_attempts):
            raw_response = await self.llm_client.chat_text(messages)
            if raw_response:
                answer = raw_response
                break
            error_message = (
                f"The last message from you was empty, perhaps you were trying to call the tool"
                f"Don't call the any tools."
            )
            messages.append({"role": "user", "content": error_message})

        if not answer:
            answer = combined_summaries
        return answer

    async def execute(self, context: PipelineContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")
        messages = self.create_messages(context)
        user_query = self.get_user_query(context)
        validation_condition = self.get_validation_condition(context)
        search_results = self.get_search_results(context)
        if not search_results:
            search_results = ['Nothing found for this request']
        combined_summaries = ",".join(search["text"] for search in search_results)

        await self._emit_stage(stage_callback, stage="sources_used", metadata={
            "sources": [{"title": src["title"], "url": src["url"]} for src in search_results]
        })

        messages.append(
            {"role": "user", "content": self.user_message(user_query, validation_condition, combined_summaries)}
        )
        answer = await self.get_answer_with_retry(messages, combined_summaries)

        result_ctx = PipelineContext.from_context(context)
        result_ctx.answer = answer
        result_ctx.sources = [f"{src["title"]}: {src['url']}" for src in search_results]
        return StepResult(context=result_ctx, stop=False)

    def stage(self) -> str:
        return "thinking"

    def create_messages(self, context) -> list[dict]:
        messages = []

        system_prompt = self.system_prompt()
        if not system_prompt:
            system_prompt = self.get_system_prompt(context)
        messages.append({"role": "system", "content": system_prompt})

        history = self.get_history(context)
        messages.extend(history)

        return messages

    @staticmethod
    def user_message(user_query: str, required_info: str, combined_summaries: str) -> str:
        return (
            f"User Query: {user_query}\n"
            f"Required Criteria: {required_info}\n\n"
            f"Extracted Information:\n{combined_summaries}\n\n"
            "Based on the extracted information above, provide a comprehensive answer to the user's query."
        )
