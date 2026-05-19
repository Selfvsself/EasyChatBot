from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.search_query_step import SearchQueryStep
from handlers.pipelines.steps.search_query_validator_step import SelectBestQueryStep
from handlers.pipelines.steps.step_context import StepContext, SearchResult
from handlers.pipelines.steps.step_result import StepResult


class SearchPlanningAgentPipeline(BaseStep):

    def __init__(self, llm_client, max_queries=3):
        super().__init__(llm_client)
        self.max_queries = max_queries
        self.web_query = SearchQueryStep(self.llm_client)
        self.web_query_validator = SelectBestQueryStep(self.llm_client)

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")
        last_context = context
        queries = []
        for _ in range(0, self.max_queries):
            query_result = await self.web_query.execute(last_context)
            if query_result.success:
                query_context = query_result.context
                search = SearchResult(query_context.next_step_query, "", query_context.next_step_query)
                queries.append(search)
        if queries:
            last_context = context
            last_context.search_results = queries
            valid_search = await self.web_query_validator.execute(last_context)
            last_context = valid_search.context

        return StepResult(context=last_context, stop=False)

    def stage(self) -> str:
        return "web_query_planning"
