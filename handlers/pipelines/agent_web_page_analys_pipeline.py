from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.search_page_source_step import SearchPageSourceStep
from handlers.pipelines.steps.search_rag_step import SearchRagStep
from handlers.pipelines.steps.search_summary_step import SearchSummaryStep
from handlers.pipelines.steps.step_context import StepContext
from handlers.pipelines.steps.step_result import StepResult


class WebPageAnalysAgentPipeline(BaseStep):

    def __init__(self, llm_client, max_pages=5, max_page_chars=20000, top_chunks=1, chunk_size=5000, chunk_overlap=500):
        super().__init__(llm_client)
        self.max_pages = max_pages
        self.max_page_chars = max_page_chars
        self.top_chunks = top_chunks
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.web_source = SearchPageSourceStep(self.llm_client, max_results=max_pages * 2,
                                               max_page_chars=self.max_page_chars)
        self.web_rag = SearchRagStep(self.llm_client, top_chunks=self.top_chunks, chunk_size=self.chunk_size,
                                     chunk_overlap=self.chunk_overlap)
        self.web_summary = SearchSummaryStep(self.llm_client)

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")
        is_success = False
        last_context = context
        search_results = last_context.search_results
        pages_body = []
        for search_result in search_results:
            await self._emit_stage(stage_callback, stage="site_reading", metadata={
                "url": search_result.url,
                "title": search_result.title})
            search_context = StepContext.from_context(last_context)
            search_context.search_results = [search_result]
            source_page_result = await self.web_source.execute(search_context)
            source_page_ctx = source_page_result.context
            actual_source_page = source_page_ctx.search_results
            if actual_source_page:
                rag_result = await self.web_rag.execute(source_page_ctx)
                rag_ctx = rag_result.context
                summary_result = await self.web_summary.execute(rag_ctx)
                summary_ctx = summary_result.context
                if summary_result.success and summary_ctx:
                    pages_body.extend(summary_ctx.search_results)
                    if len(pages_body) >= self.max_pages:
                        break
        last_context.search_results = pages_body

        return StepResult(context=last_context, stop=False, success=is_success)

    def stage(self) -> str:
        return "web_query_planning"
