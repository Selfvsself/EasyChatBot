import json
import re

from pydantic import BaseModel, ValidationError

from .base_handler import BaseHandler
from services.agent_orchestrator import AgentOrchestrator


class TranslatorHandler(BaseHandler):
    MAX_ITERATIONS = 2

    class TranslationPayload(BaseModel):
        context_notes: str | None = None
        translation: str = ""
        detected_language_code: str = "unknown"

    def __init__(self, llm_client, message_repo):
        super().__init__(llm_client, message_repo)
        self.orchestrator = AgentOrchestrator(llm_client=llm_client, max_iterations=self.MAX_ITERATIONS)

    @classmethod
    def _extract_translation(cls, raw_response: str) -> str:
        raw_text = (raw_response or "").strip()
        if not raw_text:
            return ""

        clean_json = re.sub(r"```(?:json)?\n?|```", "", raw_text).strip()
        try:
            payload = json.loads(clean_json)
            parsed = cls.TranslationPayload.model_validate(payload)
            return parsed.translation.strip()
        except (json.JSONDecodeError, ValidationError, TypeError):
            return raw_text

    @staticmethod
    def _build_messages_block(messages):
        lines = []
        for message in messages:
            role = getattr(message, "role", "user")
            if role == "system":
                continue
            text = str(getattr(message, "text", "")).strip().replace("\n", " ")
            if text:
                lines.append(f"{role}: {text}")
        return "\n".join(lines)

    async def handle(self, chat, app, text, tools, stage_callback=None):
        extracted_facts = "None"
        if chat.memory_summary:
            extracted_facts = f"\n\nChat memory (summary of archived messages):\n{chat.memory_summary.strip()}"
        history = self.message_repo.get_by_chat(chat.id, limit=50, include_archived=False)
        system_prompt = app.system_prompt.replace("[INSERT_PREVIOUS_FACTS_HERE]", extracted_facts)


        prompt = self.build_prompt(system=system_prompt, history=list(reversed(history)), text=text)
        raw_answer = await self.orchestrator.run(
            user_query=text,
            history=prompt,
            context=extracted_facts,
            tools=tools,
            stage_callback=stage_callback,
            response_format="json"
        )
        return self._extract_translation(raw_answer)
