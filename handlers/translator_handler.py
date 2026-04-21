import json
import re

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

    def parse_llm_response(self, raw_response):
        """
        Парсит JSON из ответа LLM и возвращает словарь с полями.
        """
        try:
            clean_json = re.sub(r'```(?:json)?\n?|```', '', raw_response).strip()

            data = json.loads(clean_json)

            translation = data.get("translation", "")
            lang_code = data.get("detected_language_code", "unknown")
            context_notes = data.get("context_notes", None)

            return {
                "translation": translation,
                "lang": lang_code,
                "notes": context_notes
            }

        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON: {e}")
            return None
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            return None

    async def handle(self, chat, app, text):
        history = self.message_repo.get_by_chat(chat.id, limit=20)
        facts_prompt = self.build_prompt(self.CONTEXT_EXTRACTOR_SYSTEM, list(reversed(history)), text=text)
        extracted_facts = await self.llm.chat(facts_prompt)

        system_prompt = app.system_prompt.replace("[INSERT_PREVIOUS_FACTS_HERE]", extracted_facts)
        self.build_prompt(self.CONTEXT_EXTRACTOR_SYSTEM, list(reversed(history)), text=text)
        final_prompt = self.build_prompt(system_prompt, [], text=text)

        raw_answer = await self.llm.chat(final_prompt)

        result = self.parse_llm_response(raw_answer)
        return result.get("translation")
