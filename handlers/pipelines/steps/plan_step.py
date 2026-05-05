import json
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel

from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.step_context import StepContext
from handlers.pipelines.steps.step_result import StepResult


class StepAction(StrEnum):
    SEARCH = "SEARCH"
    CLARIFY = "CLARIFY"
    RESPOND = "RESPOND"


class PlanStep(BaseStep):
    class SearchPlan(BaseModel):
        action: StepAction
        reason: str
        next_step: str
        completeness_criteria: str

    async def parse_search_plan_with_retry(self, context: StepContext) -> SearchPlan | None:
        search_plan = None
        max_attempts = 3
        system_prompt = self.system_prompt()
        history = self.get_history(context)
        user_query = self.get_user_query(context)
        internal_messages = self.get_internal_messages(context)
        messages = self.create_messages(system_prompt, history, internal_messages, user_query)
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
                action=StepAction.RESPOND,
                reason="you can answer it",
                next_step=user_query,
                completeness_criteria="user's intent is fully satisfied"
            )
        return search_plan

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")
        search_plan = await self.parse_search_plan_with_retry(context)

        internal_messages = self.get_internal_messages(context)
        internal_messages.append(
            "ROUTING AGENT\n"
            f"NEXT ACTION: {search_plan.action}\n"
            f"NEXT STEP QUERY: {search_plan.next_step}\n"
            f"COMPLETENESS CRITERIA: {search_plan.completeness_criteria}"
        )

        result_ctx = StepContext.from_context(context)
        result_ctx.action = search_plan.action
        result_ctx.next_step_query = search_plan.next_step
        result_ctx.validation_condition = search_plan.completeness_criteria
        result_ctx.internal_messages = internal_messages
        return StepResult(context=result_ctx, stop=False)

    def stage(self) -> str:
        return "web_query_planning"

    def system_prompt(self):
        current_date = datetime.now().strftime("%A, %d %B %Y")

        return (
            "You are an intelligent routing agent and a search planner.\n"
            "Your job is to analyze the incoming user query and dialogue history to either create a structured plan "
            "for retrieving information, ask for clarification, or decide to respond directly.\n\n"
            "RULES:\n"
            f"- The current date is {current_date}. Use this to determine if the query requires fresh information.\n"
            "- Do NOT answer the question directly if you decide to search or clarify.\n"
            "- Do NOT call any tools.\n"
            "- Only return valid JSON. No markdown blocks like ```json, no conversational filler, and no extra text outside the JSON.\n"
            "- The output JSON must strictly follow the schema provided below.\n"
            "- Set 'action' to:\n"
            "  * \"SEARCH\": if the user query requires up-to-date information, real-time data, or specific facts not likely to be in your training data.\n"
            "  * \"CLARIFY\": if the query is too vague, contains ambiguous terms, or lacks sufficient context to provide a useful answer.\n"
            "  * \"RESPOND\": if the query is fully understood, requires no external data, and you can answer it "
            "directly (e.g., writing code, solving math, rewriting text, or answering based on general knowledge).\n"
            "- Determine the language the user speaks and use that same language when filling in the next_step fields.\n"
            "- ALIGNMENT RULE: The 'completeness_criteria' must strictly and exclusively map to the operation described "
            "in 'next_step'. Do NOT invent additional verification steps or secondary searches that are not explicitly "
            "covered by the 'next_step' instruction. For example, if action=SEARCH, 'completeness_criteria' should only "
            "state that the specific data requested in the 'next_step' search query has been found.\n\n"
            "OUTPUT JSON SCHEMA:\n"
            "{\n"
            "  \"action\": \"SEARCH\" | \"CLARIFY\" | \"RESPOND\",\n"
            "  \"reason\": \"A brief explanation of your choice in English\",\n"
            "  \"next_step\": \"If action=SEARCH: an optimized search query. If action=CLARIFY: a clarification question to the user. If action=RESPOND: the original user query verbatim.\",\n"
            "  \"completeness_criteria\": \"Clear criteria for when the user's intent is fully satisfied. For SEARCH: what specific facts requested in the search query must be found. For RESPOND: what key points from the user query must be covered in the final answer. For CLARIFY: what specific missing detail must be obtained from the user.\"\n"
            "}"
        )
