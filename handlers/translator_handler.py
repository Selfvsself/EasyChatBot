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

    async def handle(self, chat, app, text, tools, stage_callback=None):
        history = self.message_repo.get_by_chat(chat.id, limit=20, include_archived=False)
        memory_part = ""
        if chat.memory_summary:
            memory_part = f"\n\nChat memory (summary of archived messages):\n{chat.memory_summary.strip()}"

        system_prompt = (
            f"{app.system_prompt}{memory_part}\n\n"
            "You are in translator mode. "
            "Preserve meaning, tone, and named entities accurately."
        )
        prompt = self.build_prompt(system=system_prompt, history=list(reversed(history)), text=text)
        raw_answer = await self.orchestrator.run(
            base_messages=prompt,
            chat=chat,
            app=app,
            tools=tools,
            stage_callback=stage_callback,
        )
        return self._extract_translation(raw_answer)
