from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.step_context import StepContext, SearchResult
from handlers.pipelines.steps.step_result import StepResult
from handlers.tools.web_search_tool import WebSearchTool


class SearchWebStep(BaseStep):

    def __init__(self, llm_client, max_results=5, max_page_chars=6000):
        super().__init__(llm_client)
        self.max_results = max_results
        self.web_search_tool = WebSearchTool(max_results, max_page_chars)

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")
        is_success = False
        search_results = []
        seen_urls = set()
        search_query = self.get_next_step_query(context)
        web_results = await self.web_search_tool.search(search_query)
        for web_page in web_results:
            url = web_page['url']
            if url not in seen_urls:
                seen_urls.add(url)
                search_results.append(SearchResult(web_page['title'], url, web_page['snippet']))

        result_ctx = StepContext.from_context(context)
        result_ctx.search_results = search_results
        return StepResult(context=result_ctx, stop=False, success=is_success)

    def stage(self) -> str:
        return "web_search_running"
