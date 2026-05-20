import json

from pydantic import BaseModel

from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.step_context import StepContext
from handlers.pipelines.steps.step_result import StepResult


class GenerateWebAnswerStep(BaseStep):
    class WebAnswer(BaseModel):
        final_response: str

    def parse_summary_result(self, raw):
        try:
            payload = json.loads(self._strip_fences(raw))
            return self.WebAnswer.model_validate(payload)
        except Exception:
            return self.WebAnswer(
                final_response="Something went wrong, try again later"
            )

    async def get_answer_with_retry(self, context: StepContext) -> WebAnswer | None:
        summary = None
        max_attempts = 3
        system_prompt = self.get_system_prompt(context)
        user_prompt = self.user_message(context)
        print("GenerateWebAnswerStep system_prompt:\n", system_prompt)
        print("GenerateWebAnswerStep user_prompt:\n", user_prompt)
        messages = self.create_messages(system_prompt, [], user_prompt)
        for attempt in range(max_attempts):
            try:
                raw_response = await self.llm_client.chat_json(messages)
                print("GenerateWebAnswerStep raw_response:\n", raw_response)
                messages.append({"role": "assistant", "content": raw_response})
                payload = json.loads(self._strip_fences(raw_response))
                summary = self.WebAnswer.model_validate(payload)
                break
            except Exception as e:
                error_message = (
                    f"Your previous response caused a parsing error: {str(e)}. "
                    f"Please correct the output and return ONLY valid JSON matching the required schema."
                )
                messages.append({"role": "user", "content": error_message})

        if not summary:
            summary = self.WebAnswer(
                final_response="Something went wrong, try again later"
            )
        return summary

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")

        answer = await self.get_answer_with_retry(context)
        result_ctx = StepContext.from_context(context)
        result_ctx.answer = answer.final_response

        return StepResult(context=result_ctx, stop=False)

    def user_message(self, context: StepContext) -> str:
        user_query = self.get_user_query(context)
        validation_condition = self.get_validation_condition(context)
        search_results = self.get_search_results(context)
        history = self.get_history(context)
        chat_memory = self.get_chat_memory(context)

        web_data = []
        for search_result in search_results:
            web_data.append({
                "url": search_result.url,
                "title": search_result.title,
                "content": search_result.text
            })
        user_intent = self.get_user_intent(context)
        output_data = {
            "meta": {
                "agent": "information_extractor"
            },
            "task": {
                "user_message": user_query,
                "required_criteria": validation_condition,
                "normalized_intent": user_intent
            },
            "data": {
                "search_results": web_data
            },
            "context": {
                "recent_history": history,
                "chat_memory": chat_memory
            }
        }

        return json.dumps(output_data, ensure_ascii=False, indent=2)

    def stage(self) -> str:
        return "typing"

    def stage_metadata(self) -> dict:
        return {}

    def get_system_prompt(self, context: StepContext = None):
        return self.set_prompt_templates(
            "You are a master synthesis and response agent. Your job is to analyze the user's intent, the required "
            "criteria, and the extracted web data, then generate a comprehensive, accurate, and direct final answer "
            "for the user.\n\n"
            "INPUT STRUCTURE:\n"
            "You will receive a JSON containing:\n"
            "- \"task\": Includes \"user_message\" (original text), \"required_criteria\" (mandatory "
            "filters/constraints), and \"normalized_intent\" (reconstructed global goal).\n"
            "- \"data\": \"search_results\" array, where each item contains \"url\", \"title\", and \"content\" "
            "(cleaned facts/text from relevant web pages).\n"
            "- \"context\": \"recent_history\" and \"chat_memory\".\n\n"
            "RULES FOR RESPONSE GENERATION:\n"
            "1. Direct Answer First: Start your response with the most critical information that answers the core query"
            " immediately. Do not use conversational filler or introductions.\n"
            "2. Criterion Fulfillment: Ensure every single constraint mentioned in \"required_criteria\" is thoroughly"
            " addressed using the data provided. If the data is partially missing for a specific criterion, explicitly"
            " state what is missing based only on the facts.\n"
            "3. Strict Factuality: Rely ONLY on the clear facts provided within \"search_results\". Do NOT assume, "
            "extrapolate, or invent details. If the search results do not contain the answer, politely state that the "
            "information was not found.\n"
            "4. Inline Citations: Back up every factual claim with a markdown link to the source from the "
            "\"search_results\". Use the format: [Source Title](URL). Do not group links at the bottom; anchor them "
            "naturally to the words or sentences they verify.\n"
            "5. Language & Tone: Write the final response in the language of the \"user_message\". Maintain a helpful, "
            "objective, and clear peer-to-peer tone.\n"
            "6. Formatting: Optimize for scannability. Use bullet points, short sentences, and bolding for key terms "
            "to make the text easy to scan.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a JSON object. No markdown code blocks, no extra text before or after the JSON.\n"
            "{\n"
            "  \"final_response\": \"The complete, formatted final answer to the user with inline markdown citations\"\n"
            "}"
        )
