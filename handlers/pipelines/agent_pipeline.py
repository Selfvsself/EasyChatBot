from handlers.pipelines.steps.step_result import StepResult
from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.step_context import StepContext
from handlers.pipelines.steps.plan_step import PlanStep
from handlers.pipelines.steps.enum.step_action import StepAction
from handlers.pipelines.steps.generate_step import GenerateStep
from handlers.pipelines.agent_web_search_pipeline import WebAgentPipeline
from handlers.pipelines.steps.validation_criteria_step import ValidationCriteriaStep
from handlers.pipelines.steps.search_validations_step import SearchValidationStep, ValidationAction


class AgentPipeline(BaseStep):

    def __init__(self, llm_client, max_pages = 8, max_attempt = 3):
        super().__init__(llm_client)
        self.max_pages = max_pages
        self.max_attempt = max_attempt
        self.plan_step = PlanStep(self.llm_client)
        self.answer_step = GenerateStep(self.llm_client)
        self.web_agent = WebAgentPipeline(self.llm_client, max_pages=8)
        self.web_validation = SearchValidationStep(self.llm_client)
        self.validation_criteria = ValidationCriteriaStep(self.llm_client)

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")
        last_context = context
        max_attempts = self.max_attempt
        for attempt in range(max_attempts):
            search_plan = await self.plan_step.execute(last_context)
            last_context = search_plan.context
            next_step = last_context.action
            valid_criteria = await self.validation_criteria.execute(last_context)
            last_context = valid_criteria.context
            if next_step == StepAction.SEARCH:
                web_result = await self.web_agent.execute(last_context, stage_callback)
                last_context = web_result.context
            elif next_step == StepAction.CLARIFY:
                await self._emit_stage(stage_callback, "thinking", {})
                last_context = StepContext.from_context(last_context)
                last_context.answer = last_context.next_step_query
                return StepResult(context=last_context, stop=False)
            else:
                await self._emit_stage(stage_callback, "thinking", {})
                await self.send_sources(last_context, stage_callback)
                result = await self.answer_step.execute(last_context)
                last_context = result.context
                return StepResult(context=last_context, stop=False)

            await self._emit_stage(stage_callback, "web_validating", {})
            valid_result = await self.web_validation.execute(last_context)
            if valid_result.success or attempt == max_attempts - 1:
                await self.send_sources(last_context, stage_callback)
                return StepResult(context=last_context, stop=False)
            else:
                await self._emit_stage(stage_callback, "thinking", {})
                valid_ctx = valid_result.context
                next_action = valid_ctx.action
                if next_action == ValidationAction.RESPOND:
                    answer_second_result = await self.answer_step.execute(valid_ctx)
                    answer_second_ctx = answer_second_result.context
                    valid_second_result = await self.web_validation.execute(answer_second_ctx)
                    if valid_second_result.success:
                        last_context = answer_second_ctx
                        await self.send_sources(last_context, stage_callback)
                        return StepResult(context=last_context, stop=False)
                else:
                    continue

        await self.send_sources(last_context, stage_callback)

        return StepResult(context=last_context, stop=False)

    def stage(self) -> str:
        return "thinking"

    async def send_sources(self, context: StepContext, stage_callback=None):
        sources = context.search_results
        if sources:
            await self._emit_stage(stage_callback, stage="sources_used", metadata={
                "sources": [{"title": src.title, "url": src.url} for src in sources]
            })
