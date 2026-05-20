from handlers.pipelines.agent_search_planning_pipeline import SearchPlanningAgentPipeline
from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.generate_web_answer_step import GenerateWebAnswerStep
from handlers.pipelines.steps.search_page_source_step import SearchPageSourceStep
from handlers.pipelines.steps.search_rag_step import SearchRagStep
from handlers.pipelines.steps.search_summary_step import SearchSummaryStep
from handlers.pipelines.steps.search_web_step import SearchWebStep
from handlers.pipelines.steps.step_context import StepContext
from handlers.pipelines.steps.step_result import StepResult
from handlers.pipelines.agent_web_page_analys_pipeline import WebPageAnalysAgentPipeline


class WebAgentPipeline(BaseStep):

    def __init__(self, llm_client, max_pages=5):
        super().__init__(llm_client)
        self.max_pages = max_pages
        self.answer_step = GenerateWebAnswerStep(self.llm_client)
        self.web_search = SearchWebStep(self.llm_client, max_results=max_pages * 2, max_page_chars=20000)
        self.web_source = SearchPageSourceStep(self.llm_client)
        self.web_rag = SearchRagStep(self.llm_client)
        self.web_summary = SearchSummaryStep(self.llm_client)
        self.query_plan = SearchPlanningAgentPipeline(self.llm_client)
        self.web_page_analys = WebPageAnalysAgentPipeline(self.llm_client)

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")
        last_context = context
        await self._emit_stage(stage_callback, "web_query_planning", {})
        query_plan_result = await self.query_plan.execute(last_context)
        last_context = query_plan_result.context
        await self._emit_stage(stage_callback, stage="web_search_running",
                               metadata={"query": last_context.next_step_query})
        web_result = await self.web_search.execute(last_context)
        last_context = web_result.context
        web_page_results = await self.web_page_analys.execute(last_context, stage_callback)
        last_context = web_page_results.context
        await self._emit_stage(stage_callback, "typing", {})
        answer_result = await self.answer_step.execute(last_context)
        last_context = answer_result.context

        return StepResult(context=last_context, stop=False, success=answer_result.success)


    def stage(self) -> str:
        return "web_query_planning"
