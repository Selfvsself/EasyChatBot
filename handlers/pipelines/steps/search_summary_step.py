import json

from pydantic import BaseModel

from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.step_context import StepContext, SearchResult
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
        system_prompt = self.get_system_prompt(context)
        for search in search_results:
            title = search.title
            url = search.url

            user_prompt = self.user_message(context, search)
            messages = self.create_messages(system_prompt, [], user_prompt)
            summary = await self.parse_summary_result_with_retry(messages)
            if summary.useful_for_answer:
                is_success = True
                summaries.append(SearchResult(title, url, summary.compressed_text))

        result_ctx = StepContext.from_context(context)
        result_ctx.search_results = summaries
        return StepResult(context=result_ctx, stop=False, success=is_success)

    def stage(self) -> str:
        return "thinking"

    def user_message(self, context: StepContext, search_result: SearchResult) -> str:
        user_query = self.get_user_query(context)
        validation_condition = self.get_validation_condition(context)
        user_intent = self.get_next_step_query(context)
        output_data = {
            "meta": {
                "agent": "information_extractor"
            },
            "task": {
                "user_message": user_query,
                "required_criteria": validation_condition,
                "normalized_intent": user_intent
            },
            "data": {
                "web_page": {
                    "url": search_result.url,
                    "title": search_result.title,
                    "content": search_result.text
                }
            }
        }

        return json.dumps(output_data, ensure_ascii=False, indent=2)

    def get_system_prompt(self, context: StepContext = None):
        return self.set_prompt_templates(
            "You are an information extraction assistant. Your job is to analyze raw webpage text and extract ONLY the "
            "facts and data relevant to the user's normalized intent and mandatory criteria.\n\n"
            "INPUT STRUCTURE:\n"
            "You will receive a JSON containing:\n"
            "- \"task\": Includes \"user_message\", \"required_criteria\" (constraints), and \"normalized_intent\" "
            "(global goal).\n"
            "- \"data\": \"web_page\" object containing \"url\", \"title\", and \"content\" (raw webpage text).\n"
            "RULES FOR INFORMATION EXTRACTION:\n"
            "1. Usefulness Evaluation: Analyze \"web_page.content\" carefully. Determine if this page actually contains"
            " specific facts, answers, or data that satisfy the \"normalized_intent\" and match the "
            "\"required_criteria\".\n"
            "2. Negative Case: If the page is a generic error, access denied, contains no relevant info, or fails to "
            "meet the \"required_criteria\", you MUST set \"useful_for_answer\" to false and \"compressed_text\" "
            "to \"\".\n"
            "3. Positive Case (Extraction): If the page is useful, extract ONLY the direct facts, figures, and text "
            "required to answer the query. Completely remove all navigation links, advertisements, "
            "headers/footers, and irrelevant content.\n"
            "4. Objective Role: Do NOT attempt to answer the user's query yourself. Do NOT synthesize a final response."
            " Your only job is to compress and extract raw, relevant data for the next agent in the pipeline.\n"
            "5. Language: Keep the extracted facts in the original language of the webpage text, or in the language of "
            "the \"user_message\" if it facilitates easy integration.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a JSON object. No markdown blocks, no extra text.\n"
            "{\n"
            "  \"useful_for_answer\": true | false,\n"
            "  \"compressed_text\": \"Extracted and cleaned text containing only relevant facts, or empty "
            "string if not useful\"\n"
            "}"
        )
