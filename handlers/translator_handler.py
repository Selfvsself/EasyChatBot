import json
import re

from pydantic import BaseModel, ValidationError, field_validator

from .pipelines.pipeline_factory import PipelineFactory
from .pipelines.pipeline_orchestrator import PipelineOrchestrator
from .pipelines.steps.step_context import StepContext
from .base_handler import BaseHandler


class TranslatorHandler(BaseHandler):
    MAX_ITERATIONS = 2

    class TranslationPayload(BaseModel):
        context_notes: str | None = None
        translation: str = ""
        detected_language_code: str = "unknown"

        @field_validator("translation", mode="before")
        @classmethod
        def serialize_json_to_string(cls, v):
            if isinstance(v, (dict, list)):
                return json.dumps(v, ensure_ascii=False)
            return str(v)

    def __init__(self, llm_client, message_repo):
        super().__init__(llm_client, message_repo)
        self.pipeline = PipelineFactory(llm_client=llm_client)
        self.orchestrator = PipelineOrchestrator()

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

    async def handle(self, user_query, context, tools, stage_callback=None):
        history = []
        chat_memory = "None"
        system_prompt = self.prepare_system_prompt(context, chat_memory)

        context = StepContext(
            user_input=user_query,
            history=history,
            chat_memory=chat_memory,
            system_prompt=system_prompt
        )
        pipeline = self.pipeline.simple_translate_pipeline()
        result = await self.orchestrator.run(pipeline, context, stage_callback)
        return self._extract_translation(result)
