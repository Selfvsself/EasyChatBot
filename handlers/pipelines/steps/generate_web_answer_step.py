from handlers.pipelines.steps.step_context import StepContext, SearchResult
from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.step_result import StepResult


class GenerateWebAnswerStep(BaseStep):
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
        search_results = self.get_search_results(context)
        user_prompt_with_data = self.user_message(user_query, internal_messages, search_results)

        messages = self.create_messages(system_prompt, history, [], user_prompt_with_data)
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

    @staticmethod
    def user_message(user_query: str, internal_messages: list[str], web_pages: list[SearchResult]):
        internal_message = "None"
        if internal_messages:
            internal_message = "<thought>\n"
            for idx, msg in enumerate(internal_messages):
                internal_message += f"Step {idx + 1}:\n"
                internal_message += msg
                internal_message += "\n"
            internal_message += "</thought>"

        web_data = "None"
        if web_pages:
            web_data = "<web_data>\n"
            for result in web_pages:
                web_data += f"{result.text}:\n"
            web_data += "</web_data>"

        return (
            f"User Query: {user_query}\n"
            f"History of thoughts: {internal_message}\n\n"
            f"Webpage Content:\n{web_data}"
        )

    def stage(self) -> str:
        return "typing"

    def stage_metadata(self) -> dict:
        return {}
