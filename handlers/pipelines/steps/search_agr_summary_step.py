import json
from pydantic import BaseModel, Field
from handlers.pipelines.steps.pipeline_context import PipelineContext
from handlers.pipelines.steps.pipeline_step import PipelineStep
from handlers.pipelines.steps.step_result import StepResult


class SearchSummaryStep(PipelineStep):
    # 1. Схема ответа теперь ожидает ОДИН текст и массив индексов
    class AggregatedSummaryResult(BaseModel):
        useful_site_indices: list[int] = Field(
            description="Список индексов сайтов, которые содержали полезную информацию"
        )
        aggregated_compressed_text: str = Field(
            description="Единая объединенная выжимка фактов из всех полезных сайтов. Пустая строка, если полезных сайтов нет."
        )

    async def parse_summary_result_with_retry(self, messages: list[dict]) -> AggregatedSummaryResult | None:
        summary = None
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                raw_response = await self.llm_client.chat_json(messages)
                messages.append({"role": "assistant", "content": raw_response})
                payload = json.loads(self._strip_fences(raw_response))
                summary = self.AggregatedSummaryResult.model_validate(payload)
                break
            except Exception as e:
                error_message = (
                    f"Your previous response caused a parsing error: {str(e)}. "
                    f"Please correct the output and return ONLY valid JSON matching the required schema."
                )
                messages.append({"role": "user", "content": error_message})

        if not summary:
            # Дефолтный ответ в случае ошибки
            summary = self.AggregatedSummaryResult(
                useful_site_indices=[],
                aggregated_compressed_text=""
            )
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

        # Формируем промпт
        messages.append({
            "role": "user",
            "content": self.user_message(user_query, validation_condition, search_results)
        })

        # Получаем агрегированный ответ
        batch_result = await self.parse_summary_result_with_retry(messages)

        is_success = len(batch_result.useful_site_indices) > 0
        summaries = []

        # Безопасно собираем метаданные использованных сайтов для сохранения в пайплайне
        for index in batch_result.useful_site_indices:
            if 0 <= index < len(search_results):
                site = search_results[index]

                # Защита от ошибки 'string indices must be integers':
                # Если элемент списка — строка, делаем из нее словарь-заглушку
                if isinstance(site, str):
                    site = {"title": f"Result {index}", "url": "", "text": site}

                summaries.append({
                    "title": site.get("title", f"Result {index}"),
                    "url": site.get("url", ""),
                    # Здесь мы можем либо дублировать общий текст для каждого сайта,
                    # либо оставить оригинальный текст. Запишем общий агрегированный текст.
                    "text": ""
                })

        result_ctx = PipelineContext.from_context(context)
        # Сохраняем обработанные результаты (теперь они все ведут на общий текст)
        result_ctx.search_results = summaries
        result_ctx.answer = batch_result.aggregated_compressed_text

        # Опционально: можно сохранить общий саммари прямо в контекст, если пайплайн это поддерживает
        # result_ctx.aggregated_summary = batch_result.aggregated_compressed_text

        return StepResult(context=result_ctx, stop=False, success=is_success)

    def stage(self) -> str:
        return "thinking"

    @staticmethod
    def user_message(user_query: str, required_info: str, search_results: list):
        formatted_sites = []
        for index, search in enumerate(search_results):
            # Защита от ошибки 'string indices must be integers'
            if isinstance(search, str):
                search = {"title": "Unknown Title", "url": "", "text": search}

            title = search.get("title", "No Title")
            url = search.get("url", "No URL")
            text = search.get("text", "")

            formatted_sites.append(
                f"=== SITE INDEX: {index} ===\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content:\n{text}\n"
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
            "- Analyze all provided webpages in the list\n"
            "- Identify which pages contain useful information for the user's query\n"
            "- Extract facts from all useful pages and combine them into ONE single, cohesive summary (aggregated_compressed_text)\n"
            "- Do NOT create separate summaries for each page. Merge similar facts together\n"
            "- In the useful_site_indices array, list the indices of all websites you used to create this summary\n"
            "- If NO pages are useful, return an empty array for indices and an empty string for the text\n"
            "- Do NOT answer the user's query yourself\n"
            "- Only return JSON matching the schema\n\n"

            "OUTPUT JSON SCHEMA:\n"
            "{\n"
            "  \"useful_site_indices\": [int, int, ...],\n"
            "  \"aggregated_compressed_text\": str\n"
            "}"
        )
