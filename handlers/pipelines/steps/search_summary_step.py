import json
from pydantic import BaseModel, Field
from handlers.pipelines.steps.step_context import StepContext, SearchResult
from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.step_result import StepResult


class SearchSummaryStep(BaseStep):
    class AggregatedSummaryResult(BaseModel):
        aggregated_compressed_text: str = Field()
        useful_site_indices: list[int] = Field()

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
            summary = self.AggregatedSummaryResult(
                useful_site_indices=[],
                aggregated_compressed_text=""
            )
        return summary

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")

        search_results = self.get_search_results(context)
        if not search_results:
            return StepResult(context=context, stop=False, success=False)

        system_prompt = self.get_system_prompt(context)
        history = self.get_history(context)
        user_query = self.get_user_query(context)
        validation_condition = self.get_validation_condition(context)
        user_message = self.user_message(user_query, validation_condition, search_results)
        internal_messages = self.get_internal_messages(context)
        messages = self.create_messages(system_prompt, history, internal_messages, user_message)
        batch_result = await self.parse_summary_result_with_retry(messages)

        is_success = len(batch_result.useful_site_indices) > 0
        summaries = []
        result_pages_data = ""
        useful_by_url = {}

        for index in batch_result.useful_site_indices:
            if 0 <= index < len(search_results):
                site = search_results[index]
                if site.url not in useful_by_url:
                    useful_by_url[site.url] = SearchResult(site.title, site.url, "")
                    result_pages_data += f" - Title: {site.title}, URL: {site.url}\n"

        summaries.extend(useful_by_url.values())

        result_ctx = StepContext.from_context(context)
        result_ctx.search_results = summaries
        result_ctx.next_step_query = batch_result.aggregated_compressed_text

        internal_messages = self.get_internal_messages(context)
        internal_messages.append(
            "WEB SEARCH AGENT\n"
            f"SOURCES USED:\n{result_pages_data}\n"
            f"COMPRESSED TEXT OF PAGES: {batch_result.aggregated_compressed_text}\n"
        )
        result_ctx.internal_messages = internal_messages

        return StepResult(context=result_ctx, stop=False, success=is_success)

    def stage(self) -> str:
        return "thinking"

    @staticmethod
    def user_message(user_query: str, required_info: str, search_results: list[SearchResult]):
        formatted_sites = []
        for index, search in enumerate(search_results):
            title = search.title
            url = search.url
            text = search.text

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

    def get_system_prompt(self, context: StepContext = None):
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
            "  \"aggregated_compressed_text\": str,\n"
            "  \"useful_site_indices\": [int, int, ...]\n"
            "}"
        )
