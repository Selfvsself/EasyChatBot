import json
from typing import Any, Coroutine

from pydantic import BaseModel

from handlers.pipelines.steps.pipeline_context import PipelineContext
from handlers.pipelines.steps.pipeline_step import PipelineStep
from handlers.pipelines.steps.step_result import StepResult


class SearchPlanStep(PipelineStep):
    class SearchPlan(BaseModel):
        web_search_required: bool
        intent: str
        required_info: list[str]
        search_queries: list[str]
        strategy: str
        stop_condition: str
        max_searches: int = 2

    async def parse_search_plan_with_retry(self, context: PipelineContext) -> SearchPlan | None:
        search_plan = None
        max_attempts = 3
        messages = self.create_messages(context)
        for attempt in range(max_attempts):
            try:
                raw_response = await self.llm_client.chat_json(messages)
                messages.append({"role": "assistant", "content": raw_response})
                payload = json.loads(self._strip_fences(raw_response))
                search_plan = self.SearchPlan.model_validate(payload)
                break
            except Exception as e:
                error_message = (
                    f"Your previous response caused a parsing error: {str(e)}. "
                    f"Please correct the output and return ONLY valid JSON matching the required schema."
                )
                messages.append({"role": "user", "content": error_message})

        if not search_plan:
            user_query = self.get_user_query(context)
            search_plan = self.SearchPlan(
                web_search_required=False,
                intent="generic search",
                required_info=[],
                search_queries=[user_query],
                max_searches=1
            )
        return search_plan

    async def execute(self, context: PipelineContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")
        search_plan = await self.parse_search_plan_with_retry(context)

        result_ctx = PipelineContext.from_context(context)
        if search_plan.web_search_required:
            result_ctx.intent = search_plan.intent
            result_ctx.validation_condition = search_plan.stop_condition
            result_ctx.search_queries = search_plan.search_queries
        return StepResult(context=result_ctx, stop=False)

    def stage(self) -> str:
        return "web_query_planning"

    def system_prompt(self):
        return (
            "You are a search planner.\n"
            "Your job is to create a structured plan for retrieving information.\n\n"

            "RULES:\n"
            "- Do NOT answer the question\n"
            "- Do NOT call any tools\n"
            "- Only return JSON\n"
            "- Set 'web_search_required' to true only if the user query requires up-to-date information, real-time data, or specific facts not likely to be in your training data\n"
            "- Don't add details to the plan that weren't asked for\n"
            "- Don't specify what wasn't asked for\n"
            "- The simpler the request, the simpler the answer\n"
            "- The search must be done in the user's language\n"
            "- Limit search queries maximum 2\n"
            "- Prefer broad, aggregated queries\n"
            "- Avoid per-entity searches\n"
            "- Create search queries in the language of the user's last message\n\n"

            "OUTPUT JSON SCHEMA:\n"
            "{\n"
            "  \"web_search_required\": bool\n"
            "  \"intent\": str,\n"
            "  \"required_info\": list[str],\n"
            "  \"search_queries\": list[str],\n"
            "  \"strategy\": str,\n"
            "  \"stop_condition\": str,\n"
            "  \"max_searches\": int,\n"
            "}"
        )