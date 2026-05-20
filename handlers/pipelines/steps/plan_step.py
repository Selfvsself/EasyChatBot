import json

from pydantic import BaseModel

from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.enum.step_action import StepAction
from handlers.pipelines.steps.step_context import StepContext
from handlers.pipelines.steps.step_result import StepResult


class PlanStep(BaseStep):
    class RouterDecision(BaseModel):
        action: StepAction
        reason: str

    async def parse_decision_with_retry(self, context: StepContext) -> RouterDecision:
        max_attempts = 3
        system_prompt = self.get_system_prompt(context)
        history = self.get_history(context)
        user_query = self.get_user_query(context)
        validation_info = self.get_validation_condition(context)
        internal_messages = self.get_internal_messages(context)
        user_prompt = self.user_message(user_query, validation_info, internal_messages, history)
        messages = self.create_messages(system_prompt, [], [], user_prompt)

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

        result_ctx = StepContext.from_context(context)
        result_ctx.action = decision.action

        return StepResult(context=result_ctx, stop=False)

    @staticmethod
    def user_message(user_query: str, required_info: str, internal_messages: list[str], history: list[dict]) -> str:
        output_data = {
            "meta": {
                "agent": "search_planner"
            },
            "task": {
                "user_message": user_query,
                "required_criteria": required_info
            },
            "context": {
                "recent_history": history,
                "chat_memory": internal_messages
            }
        }

        return json.dumps(output_data, ensure_ascii=False, indent=2)

    def stage(self) -> str:
        return "thinking"

    def get_system_prompt(self, context: StepContext = None):
        return self.set_prompt_templates(
            "You are a routing agent. Your ONLY job is to analyze the input data and decide the next action for the "
            "user query, considering the chat history and memory.\n\n"
            "Context: Current date is ${current_date}.\n\n"
            "INPUT STRUCTURE:\n"
            "You will receive a JSON containing:\n"
            "- \"task\": The current user message and required criteria\n"
            "- \"context\": \"recent_history\" (last messages) and \"chat_memory\" (long-term facts).\n\n"
            "ACTIONS:\n"
            "1. SEARCH: Use this if the \"user_intent\" requires fresh info, news, real-time data, schedules, "
            "or specific facts.\n"
            "2. CLARIFY: Use this ONLY if both the message and the context are too vague to understand the user's goal.\n"
            "3. RESPOND: Use this if the \"user_intent\" can be answered immediately using general knowledge, "
            "logic, code, or existing context.\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a JSON object. No markdown blocks, no extra text.\n"
            "{\n"
            "\"reason\": \"Short explanation of the choice in English\",\n"
            "\"action\": \"SEARCH\" | \"CLARIFY\" | \"RESPOND\"\n"
            "}"
        )
