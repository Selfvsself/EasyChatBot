from handlers.pipelines.steps.step_result import StepResult
from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.step_context import StepContext
from handlers.pipelines.steps.plan_step import PlanStep, StepAction
from handlers.pipelines.steps.generate_step import GenerateStep
from handlers.pipelines.steps.search_web_step import SearchWebStep
from handlers.pipelines.steps.search_page_source_step import SearchPageSourceStep
from handlers.pipelines.steps.search_summary_step import SearchSummaryStep
from handlers.pipelines.steps.search_validations_step import SearchValidationStep, ValidationAction


class AgentPipeline(BaseStep):

    def __init__(self, llm_client, max_pages = 5):
        super().__init__(llm_client)
        self.max_pages = max_pages
        self.plan_step = PlanStep(self.llm_client)
        self.answer_step = GenerateStep(self.llm_client)
        self.web_search = SearchWebStep(self.llm_client, max_results=20)
        self.web_source = SearchPageSourceStep(self.llm_client)
        self.web_summary = SearchSummaryStep(self.llm_client)
        self.web_validation = SearchValidationStep(self.llm_client)

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")
        last_context = context
        max_attempts = 3
        is_validate = False
        for attempt in range(max_attempts):
            search_plan = await self.plan_step.execute(last_context)
            last_context = search_plan.context
            next_step = last_context.action
            if next_step == StepAction.SEARCH:
                await self._emit_stage(stage_callback, "web_query_planning", {})
                await self._emit_stage(stage_callback, stage="web_search_running", metadata={"query": last_context.next_step_query})
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
                            source_ctx = StepContext.from_context(last_context)
                            source_ctx.search_results = source_pages
                            await self._emit_stage(stage_callback, "thinking", {})
                            summary_result = await self.web_summary.execute(source_ctx)
                            source_pages = []
                            summary_ctx = summary_result.context
                            if not summary_ctx:
                                continue
                            answer_result = await self.answer_step.execute(summary_ctx)
                            answer_ctx = answer_result.context
                            await self._emit_stage(stage_callback, "web_validating", {})
                            valid_result = await self.web_validation.execute(answer_ctx)
                            if valid_result.success or attempt == max_attempts - 1:
                                last_context = answer_ctx
                                is_validate = True
                                break
                            else:
                                await self._emit_stage(stage_callback, "thinking", {})
                                valid_ctx = valid_result.context
                                next_action = valid_ctx.action
                                if next_action == ValidationAction.RESPOND:
                                    answer_second_result = await self.answer_step.execute(summary_result.context)
                                    answer_second_ctx = answer_second_result.context
                                    valid_second_result = await self.web_validation.execute(answer_second_ctx)
                                    if valid_second_result.success:
                                        last_context = answer_second_ctx
                                        is_validate = True
                                        break
                                elif next_action == ValidationAction.SEARCH:
                                    break
                                elif next_action == ValidationAction.CLARIFY:
                                    continue


            elif next_step == StepAction.CLARIFY:
                await self._emit_stage(stage_callback, "thinking", {})
                result_ctx = StepContext.from_context(last_context)
                result_ctx.answer = result_ctx.next_step_query
                last_context = result_ctx
                is_validate = True
            else:
                await self._emit_stage(stage_callback, "thinking", {})
                result = await self.answer_step.execute(last_context)
                result_ctx = result.context
                last_context = result_ctx
                is_validate = True

            if is_validate:
                break

        return StepResult(context=last_context, stop=False)

    def stage(self) -> str:
        return "thinking"
