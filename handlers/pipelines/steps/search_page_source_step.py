from handlers.pipelines.steps.pipeline_context import PipelineContext
from handlers.pipelines.steps.pipeline_step import PipelineStep
from handlers.pipelines.steps.step_result import StepResult
from handlers.tools.web_search_tool import WebSearchTool


class SearchPageSourceStep(PipelineStep):

    def __init__(self, llm_client, max_results=5, max_page_chars=6000):
        super().__init__(llm_client)
        self.max_results = max_results
        self.web_search_tool = WebSearchTool(max_results, max_page_chars)

    async def execute(self, context: PipelineContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")
        is_success = False
        page_sources = []
        search_results = self.get_search_results(context)
        for web_page in search_results:
            await self._emit_stage(stage_callback, stage="site_reading", metadata={
                "url": web_page["url"],
                "title": web_page["title"]})
            page_source = await self.web_search_tool.get_page_source(web_page)
            if page_source:
                page_sources.append(page_source)
                is_success = True
            if len(page_sources) >= self.max_results:
                break

        result_ctx = PipelineContext.from_context(context)
        result_ctx.search_results = page_sources
        return StepResult(context=result_ctx, stop=False, success=is_success)

    def stage(self) -> str:
        return "web_query_planning"
