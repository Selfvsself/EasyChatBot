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
        user_intent: str

    async def parse_decision_with_retry(self, context: StepContext) -> RouterDecision:
        max_attempts = 3
        system_prompt = self.get_system_prompt(context)
        user_prompt = self.user_message(context)
        print("PlanStep system_prompt:\n", system_prompt)
        print("PlanStep user_prompt:\n", user_prompt)
        messages = self.create_messages(system_prompt, [], [], user_prompt)

        decision = None
        for attempt in range(max_attempts):
            try:
                raw_response = await self.llm_client.chat_json(messages)
                print("PlanStep raw_response:\n", raw_response)
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
        result_ctx.next_step_query = decision.user_intent

        return StepResult(context=result_ctx, stop=False)

    def user_message(self, context: StepContext) -> str:
        history = self.get_history(context)
        user_query = self.get_user_query(context)
        validation_condition = self.get_validation_condition(context)
        user_intent = self.get_next_step_query(context)
        chat_memory = self.get_chat_memory(context)
        output_data = {
            "meta": {
                "agent": "next_step_planner"
            },
            "task": {
                "user_message": user_query,
                "required_criteria": validation_condition,
                "normalized_intent": user_intent
            },
            "context": {
                "recent_history": history,
                "chat_memory": chat_memory
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
            "- \"task\": Includes \"user_message\" (current text), \"required_criteria\" (constraints), and "
            "\"normalized_intent\" (reconstructed global goal).\n"
            "- \"context\": \"recent_history\" and \"chat_memory\"."
            "CRITICAL INSTRUCTION FOR INTENT:\n"
            "First, determine the true \"user_intent\". If the current user message is incomplete, short, or uses "
            "pronouns, reconstruct the full, explicit request by combining it with the \"recent_history\" and \"chat_"
            "memory\". Always write the intent in the language of the user query as a clear, standalone command.\n\n"
            "ACTIONS:\n"
            "1. SEARCH: Use this if the \"user_intent\" requires fresh info, news, real-time data, schedules, "
            "or specific facts.\n"
            "2. CLARIFY: Use this ONLY if both the message and the context are too vague to understand the user's goal.\n"
            "3. RESPOND: Use this if the \"user_intent\" can be answered immediately using general knowledge, "
            "logic, code, or existing context.\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a JSON object. No markdown blocks, no extra text.\n"
            "{\n"
            "\"user_intent\": \"Explicit reconstructed user request in the language of the user query\",\n"
            "\"reason\": \"Short explanation of the choice in English\",\n"
            "\"action\": \"SEARCH\" | \"CLARIFY\" | \"RESPOND\"\n"
            "}"
        )
