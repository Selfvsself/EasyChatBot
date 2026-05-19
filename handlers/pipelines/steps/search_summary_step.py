import json

from pydantic import BaseModel

from handlers.pipelines.steps.step_context import StepContext, SearchResult
from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.step_result import StepResult


class SearchSummaryStep(BaseStep):
    class SummaryResult(BaseModel):
        useful_for_answer: bool
        compressed_text: str

    def parse_summary_result(self, raw):
        try:
            payload = json.loads(self._strip_fences(raw))
            return self.SummaryResult.model_validate(payload)
        except Exception:
            return self.SummaryResult(
                useful_for_answer=False,
                compressed_text=""
            )

    async def parse_summary_result_with_retry(self, messages: list[dict]) -> SummaryResult | None:
        summary = None
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                raw_response = await self.llm_client.chat_json(messages)
                messages.append({"role": "assistant", "content": raw_response})
                payload = json.loads(self._strip_fences(raw_response))
                summary = self.SummaryResult.model_validate(payload)
                break
            except Exception as e:
                error_message = (
                    f"Your previous response caused a parsing error: {str(e)}. "
                    f"Please correct the output and return ONLY valid JSON matching the required schema."
                )
                messages.append({"role": "user", "content": error_message})

        if not summary:
            summary = self.SummaryResult(
                useful_for_answer=False,
                compressed_text=""
            )
        return summary

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")
        is_success = False
        summaries = []
        search_results = self.get_search_results(context)
        validation_condition = self.get_validation_condition(context)
        system_prompt = self.get_system_prompt(context)
        internal_msg = self.get_internal_messages(context)
        user_query = self.get_user_query(context)
        history = self.get_history(context)
        for search in search_results:
            title = search.title
            url = search.url
            text = search.text

            user_prompt = self.user_message(user_query, validation_condition, text, internal_msg, history)
            messages = self.create_messages(system_prompt, [], [], user_prompt)
            summary = await self.parse_summary_result_with_retry(messages)
            if summary.useful_for_answer:
                is_success = True
                summaries.append(SearchResult(title, url, summary.compressed_text))

        result_ctx = StepContext.from_context(context)
        result_ctx.search_results = summaries
        return StepResult(context=result_ctx, stop=False, success=is_success)

    def stage(self) -> str:
        return "thinking"

    @staticmethod
    def user_message(user_query: str, required_info: str, page_text: str, internal_messages: list[str], history: list[dict]):
        internal_message = "None"
        if internal_messages:
            internal_message = "<thought>\n"
            for idx, msg in enumerate(internal_messages):
                internal_message += f"Step {idx + 1}:\n"
                internal_message += msg
                internal_message += "\n"
            internal_message += "</thought>"

        history_text = "\n".join(f"- Role: '{m["role"]}' Content: '{m["content"]}'" for m in history)

        return (
            f"Conversation history: \n<history>\n{history_text}\n</history>\n\n"
            f"User Query: {user_query}\n"
            f"Required Criteria: {required_info}\n\n"
            f"History of thoughts: {internal_message}\n\n"
            f"Webpage Content:\n{page_text}"
        )

    def get_system_prompt(self, context: StepContext = None):
        return self.set_prompt_templates(
            "You are an information extraction assistant.\n"
            "Your job is to analyze the webpage text and extract only the information relevant to the user's query and criteria.\n\n"

            "RULES:\n"
            "- Analyze the provided webpage text carefully\n"
            "- Determine if the page contains answers to the requested criteria\n"
            "- If the page is NOT useful, set useful_for_answer to false and compressed_text to \"\"\n"
            "- If it IS useful, extract ONLY the facts needed for the answer. Remove ads, navigation, and fluff\n"
            "- Do NOT answer the user's query yourself\n"
            "- Only return JSON matching the schema\n\n"

            "OUTPUT JSON SCHEMA:\n"
            "{\n"
            "  \"useful_for_answer\": bool,\n"
            "  \"compressed_text\": str\n"
            "}"
        )
