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
        user_prompt = self.user_message(context)
        messages = self.create_messages(system_prompt, [], user_prompt)

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

    def user_message(self, context: StepContext) -> str:
        history = self.get_history(context)
        user_query = self.get_user_query(context)
        validation_condition = self.get_validation_condition(context)
        user_intent = self.get_user_intent(context)
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
                "recent_history": history
            }
        }

        return json.dumps(output_data, ensure_ascii=False, indent=2)

    def stage(self) -> str:
        return "thinking"

    def get_system_prompt(self, context: StepContext = None):
        return self.set_prompt_templates(
            "You are a routing agent.\n\n"
            "Your ONLY job is to decide whether the system should:\n"
            "- SEARCH for external information,\n"
            "- CLARIFY the request,\n"
            "- or RESPOND directly.\n\n"
            "You MUST be biased toward SEARCH whenever external information could improve correctness, freshness, completeness, troubleshooting quality, or confidence.\n\n"
            "Current date: ${current_date}\n\n"
            "INPUT STRUCTURE:\n"
            "You receive a JSON object with:\n"
            "- \"task\":\n"
            "  - \"user_message\"\n"
            "  - \"required_criteria\"\n"
            "  - \"normalized_intent\"\n"
            "- \"context\":\n"
            "  - \"recent_history\"\n"
            "  - \"chat_memory\"\n\n"
            "INTENT RECONSTRUCTION:\n"
            "First reconstruct the real user intent.\n\n"
            "If the latest message is short, ambiguous, contains pronouns, or depends on previous messages, combine:\n"
            "- current message\n"
            "- recent_history\n"
            "- chat_memory\n"
            "- normalized_intent\n\n"
            "Write the reconstructed intent mentally as a full standalone request in the user's language.\n\n"
            "IMPORTANT SEARCH BIAS:\n"
            "Prefer SEARCH by default.\n\n"
            "Use SEARCH whenever:\n"
            "- the user explicitly asks to search, check, verify, investigate, find, look up, analyze, troubleshoot, compare, or validate;\n"
            "- the request may benefit from recent, external, community, technical, or factual information;\n"
            "- the request involves bugs, errors, failures, debugging, compatibility issues, stack traces, unexpected behavior, configuration problems, or performance issues;\n"
            "- the request asks for causes/reasons of a problem;\n"
            "- the answer could depend on versions, updates, APIs, libraries, frameworks, products, pricing, policies, news, or changing information;\n"
            "- external information would significantly improve confidence or accuracy;\n"
            "- there is uncertainty and web search could reduce hallucination risk;\n"
            "- the user asks for examples, references, best practices, recommendations, discussions, benchmarks, or real-world experiences.\n\n"
            "IMPORTANT:\n"
            "If there is ANY reasonable doubt whether SEARCH is needed, choose SEARCH.\n\n"
            "RESPOND should be used ONLY when:\n"
            "- the answer can be produced confidently from stable general knowledge;\n"
            "- no external information would meaningfully improve the answer;\n"
            "- the task is pure reasoning, writing, transformation, summarization, coding syntax, or explanation;\n"
            "- the user explicitly requests no search.\n\n"
            "CLARIFY should be used ONLY when:\n"
            "- the user's goal cannot be determined even after using history and memory;\n"
            "- critical information is missing and searching would likely be useless.\n\n"
            "NEVER avoid SEARCH just because you can guess a possible answer.\n\n"
            "OUTPUT:\n"
            "Return ONLY valid JSON.\n"
            "{\n"
            "  \"reason\": \"Short explanation in English\",\n"
            "  \"action\": \"SEARCH\" | \"CLARIFY\" | \"RESPOND\""
            "}"
        )
