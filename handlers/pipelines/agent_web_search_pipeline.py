from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.generate_step import GenerateStep
from handlers.pipelines.steps.search_page_source_step import SearchPageSourceStep
from handlers.pipelines.steps.search_rag_step import SearchRagStep
from handlers.pipelines.steps.search_summary_step import SearchSummaryStep
from handlers.pipelines.steps.search_query_step import SearchQueryStep
from handlers.pipelines.steps.search_web_step import SearchWebStep
from handlers.pipelines.steps.step_context import StepContext
from handlers.pipelines.steps.step_result import StepResult


class WebAgentPipeline(BaseStep):

    def __init__(self, llm_client, max_pages=5):
        super().__init__(llm_client)
        self.max_pages = max_pages
        self.answer_step = GenerateStep(self.llm_client)
        self.web_search = SearchWebStep(self.llm_client, max_results=max_pages * 2, max_page_chars=12000)
        self.web_source = SearchPageSourceStep(self.llm_client)
        self.web_rag = SearchRagStep(self.llm_client)
        self.web_summary = SearchSummaryStep(self.llm_client)
        self.web_query = SearchQueryStep(self.llm_client)

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")
        last_context = context
        await self._emit_stage(stage_callback, "web_query_planning", {})
        query_result = await self.web_query.execute(last_context)
        last_context = query_result.context
        await self._emit_stage(stage_callback, stage="web_search_running",
                               metadata={"query": last_context.next_step_query})
        result = await self.web_search.execute(last_context)
        last_context = result.context
        search_results = last_context.search_results
        source_pages = []
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
                source_pages.extend(actual_source_page)
                if len(source_pages) >= self.max_pages:
                    break
        source_ctx = StepContext.from_context(last_context)
        source_ctx.search_results = source_pages
        await self._emit_stage(stage_callback, "thinking", {})
        rag_result = await self.web_rag.execute(source_ctx)
        rag_ctx = rag_result.context
        summary_result = await self.web_summary.execute(rag_ctx)
        summary_ctx = summary_result.context
        if summary_ctx:
            last_context = summary_ctx
        answer_result = await self.answer_step.execute(last_context)
        last_context = answer_result.context

        return StepResult(context=last_context, stop=False)


def stage(self) -> str:
    return "web_query_planning"
