import json
from datetime import datetime

from pydantic import BaseModel

from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.step_context import StepContext
from handlers.pipelines.steps.step_result import StepResult
from handlers.pipelines.steps.enum.step_action import StepAction


class PlanStep(BaseStep):
    class RouterDecision(BaseModel):
        action: StepAction
        reason: str

    async def parse_decision_with_retry(self, context: StepContext) -> RouterDecision:
        max_attempts = 3
        system_prompt = self.get_system_prompt(context)
        history = self.get_history(context)
        user_query = self.get_user_query(context)
        internal_messages = self.get_internal_messages(context)
        messages = self.create_messages(system_prompt, history, internal_messages, user_query)

        decision = None
        for attempt in range(max_attempts):
            try:
                raw_response = await self.llm_client.chat_json(messages)
                payload = json.loads(self._strip_fences(raw_response))
                decision = self.RouterDecision.model_validate(payload)
                break
            except Exception as e:
                messages.append({"role": "assistant", "content": raw_response})
                error_message = f"Invalid JSON or schema: {str(e)}. Return ONLY JSON with 'action' and 'reason'."
                messages.append({"role": "user", "content": error_message})

        if not decision:
            decision = self.RouterDecision(action=StepAction.RESPOND, reason="fallback to direct response")

        return decision

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")

        decision = await self.parse_decision_with_retry(context)

        internal_messages = self.get_internal_messages(context)
        internal_messages.append(f"ROUTER DECISION: {decision.action} (Reason: {decision.reason})")

        result_ctx = StepContext.from_context(context)
        result_ctx.action = decision.action
        result_ctx.internal_messages = internal_messages

        return StepResult(context=result_ctx, stop=False)

    def stage(self) -> str:
        return "thinking"

    def get_system_prompt(self, context: StepContext = None):
        current_date = datetime.now().strftime("%A, %d %B %Y")
        return (
            "You are a routing agent. Your ONLY job is to decide the next action for the user query.\n\n"
            f"Context: Current date is {current_date}.\n\n"
            "ACTIONS:\n"
            "1. SEARCH: Use this if the query needs fresh info, news, real-time data, or specific facts you don't know.\n"
            "2. CLARIFY: Use this if the query is too vague or ambiguous to act upon.\n"
            "3. RESPOND: Use this if you can answer immediately (general knowledge, creative writing, code, math).\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            "  \"reason\": \"short explanation in English\",\n"
            "  \"action\": \"SEARCH\" | \"CLARIFY\" | \"RESPOND\"\n"
            "}"
        )
