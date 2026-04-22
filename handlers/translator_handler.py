import json
import re

from pydantic import BaseModel, ValidationError

from .base_handler import BaseHandler


class TranslatorHandler(BaseHandler):
    CONTEXT_EXTRACTOR_SYSTEM = """Role:
    You are a Context Analysis Assistant. Your goal is to extract key facts and background information from a conversation history to help a translator provide a more accurate and context-aware translation of the user's latest message.
    Input:
    Conversation History: A list of previous messages.
    Current Message: The text the user wants to translate right now.
    Instructions:
    Identify Entities: Extract names of people, project titles, specific tools, or software mentioned in the history that appear relevant to the current message.
    Determine Relationships: Identify who the user is talking to (e.g., a manager, a subordinate, a client) to help set the right level of formality.
    Summarize Topics: Briefly state the main subject of the ongoing discussion (e.g., "Discussing the UI bug in the login screen").
    Be Concise: Provide only the facts. Do not add interpretations or conversational filler.
    Relevance Filter: If the history contains no information relevant to the current message, return an empty summary or "No relevant context found."
    Output Format:
    A short, bulleted list of facts in English.
    Example Output:
    Project: "Lunar Alpha" API integration.
    Colleague: Sarah (Frontend Lead).
    Current status: Delay in backend response.
    Tone: Explaining a technical issue to a peer.
    """

    class TranslationPayload(BaseModel):
        translation: str = ""
        detected_language_code: str = "unknown"
        context_notes: str | None = None

    def parse_llm_response(self, raw_response: str):
        fallback = {
            "translation": (raw_response or "").strip(),
            "lang": "unknown",
            "notes": None,
        }

        try:
            clean_json = re.sub(r"```(?:json)?\n?|```", "", raw_response or "").strip()
            data = json.loads(clean_json)
            parsed = self.TranslationPayload.model_validate(data)
            return {
                "translation": parsed.translation,
                "lang": parsed.detected_language_code,
                "notes": parsed.context_notes,
            }
        except (json.JSONDecodeError, ValidationError, TypeError):
            return fallback

    async def handle(self, chat, app, text, tools):
        history = self.message_repo.get_by_chat(chat.id, limit=20)
        facts_prompt = self.build_prompt(self.CONTEXT_EXTRACTOR_SYSTEM, list(reversed(history)), text=text)
        extracted_facts = await self.llm.chat(facts_prompt)

        system_prompt = (
            app.system_prompt.replace("[INSERT_PREVIOUS_FACTS_HERE]", extracted_facts)
            + "\n\nReturn valid JSON only with keys: translation, detected_language_code, context_notes."
        )
        final_prompt = self.build_prompt(system_prompt, [], text=text)

        raw_answer = await self.llm.chat(final_prompt)
        result = self.parse_llm_response(raw_answer)
        return result["translation"]
