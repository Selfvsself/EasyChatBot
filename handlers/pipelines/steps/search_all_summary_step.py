import json
from pydantic import BaseModel, Field
from handlers.pipelines.steps.pipeline_context import PipelineContext
from handlers.pipelines.steps.pipeline_step import PipelineStep
from handlers.pipelines.steps.step_result import StepResult


class SearchSummaryStep(PipelineStep):
    # 1. Обновляем схемы данных для работы со списком
    class SiteSummary(BaseModel):
        site_index: int = Field(description="Индекс сайта из входящего списка")
        useful_for_answer: bool = Field(description="Полезен ли сайт для ответа")
        compressed_text: str = Field(description="Сжатый релевантный текст или пустая строка")

    class BatchSummaryResult(BaseModel):
        results: list["SearchSummaryStep.SiteSummary"]

    async def parse_summary_result_with_retry(self, messages: list[dict]) -> BatchSummaryResult | None:
        summary = None
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                raw_response = await self.llm_client.chat_json(messages)
                messages.append({"role": "assistant", "content": raw_response})
                payload = json.loads(self._strip_fences(raw_response))
                summary = self.BatchSummaryResult.model_validate(payload)
                break
            except Exception as e:
                error_message = (
                    f"Your previous response caused a parsing error: {str(e)}. "
                    f"Please correct the output and return ONLY valid JSON matching the required schema."
                )
                messages.append({"role": "user", "content": error_message})

        if not summary:
            summary = self.BatchSummaryResult(results=[])
        return summary

    async def execute(self, context: PipelineContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")

        search_results = self.get_search_results(context)
        if not search_results:
            return StepResult(context=context, stop=False, success=False)

        messages = [{"role": "system", "content": self.system_prompt()}]
        suggestions_message = self.get_suggestions_message(context)
        if suggestions_message:
            messages.append(suggestions_message)

        user_query = self.get_user_query(context)
        validation_condition = self.get_validation_condition(context)

        messages.append({
            "role": "user",
            "content": self.user_message(user_query, validation_condition, search_results)
        })

        batch_result = await self.parse_summary_result_with_retry(messages)

        summaries = []
        is_success = False

        results_by_index = {res.site_index: res for res in batch_result.results}

        for index, search in enumerate(search_results):
            llm_res = results_by_index.get(index)

            if llm_res and llm_res.useful_for_answer:
                is_success = True
                summaries.append({
                    "title": search["title"],
                    "url": search["url"],
                    "text": llm_res.compressed_text
                })

        result_ctx = PipelineContext.from_context(context)
        result_ctx.search_results = summaries
        return StepResult(context=result_ctx, stop=False, success=is_success)

    def stage(self) -> str:
        return "thinking"

    @staticmethod
    def user_message(user_query: str, required_info: str, search_results: list[dict]):
        formatted_sites = []
        for index, search in enumerate(search_results):
            formatted_sites.append(
                f"=== SITE INDEX: {index} ===\n"
                f"Title: {search['title']}\n"
                f"URL: {search['url']}\n"
                f"Content:\n{search['text']}\n"
                f"===========================\n"
            )

        sites_text = "\n".join(formatted_sites)

        return (
            f"User Query: {user_query}\n"
            f"Required Criteria: {required_info}\n\n"
            f"Webpages List:\n"
            f"{sites_text}"
        )

    def system_prompt(self):
        return (
            "You are an information extraction assistant.\n"
            "Your job is to analyze the list of webpage texts and extract only the information relevant to the user's query and criteria.\n\n"

            "RULES:\n"
            "- Analyze each provided webpage text in the list carefully\n"
            "- For each webpage, determine if it contains answers to the requested criteria\n"
            "- If the page is NOT useful, set useful_for_answer to false and compressed_text to \"\"\n"
            "- If it IS useful, extract ONLY the facts needed for the answer. Remove ads, navigation, and fluff\n"
            "- Do NOT answer the user's query yourself\n"
            "- You must process ALL sites provided in the user message\n"
            "- Only return JSON matching the schema\n\n"

            "OUTPUT JSON SCHEMA:\n"
            "{\n"
            "  \"results\": [\n"
            "    {\n"
            "      \"site_index\": int,\n"
            "      \"useful_for_answer\": bool,\n"
            "      \"compressed_text\": str\n"
            "    }\n"
            "  ]\n"
            "}"
        )
