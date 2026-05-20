from abc import abstractmethod
from types import SimpleNamespace

from .tools.base_tool import BaseTool


class BaseHandler:

    def __init__(self, llm_client, message_repo):
        self.llm = llm_client
        self.message_repo = message_repo

    @staticmethod
    def _to_dict_history(history):
        converted = []
        for message in history:
            role = getattr(message, "role", "user")
            content = str(getattr(message, "text", ""))

            converted.append({"role": role, "content": content})

        return converted

    def clean_history(self, history, text):
        """
        Удаляет последнее сообщение в истории, если оно дублирует текущий ввод (text).

        Это предотвращает зацикливание или повторную обработку одного и того же запроса.
        """

        if not history:
            history = []

        last_message = history[-1]

        if (hasattr(last_message, 'role') and last_message.role.strip() == "user" and
                hasattr(last_message, 'text') and last_message.text.strip() == text.strip()):
            history.pop()

        return self._to_dict_history(history)

    def prepare_chat_memory(self, context):
        """
        Проверяет и подготавливает chat_memory
        """

        chat_memory = context.chat_memory
        if not chat_memory:
            chat_memory = "None"
        return chat_memory

    def prepare_system_prompt(self, context, chat_memory):
        """
        Проверяет и подготавливает system_prompt
        """

        system_prompt = context.system_prompt
        if not system_prompt:
            raise ValueError("System prompt is missing or empty")
        if not chat_memory:
            raise ValueError("Chat id is missing or empty")
        system_prompt = system_prompt.replace("[INSERT_PREVIOUS_FACTS_HERE]", chat_memory)
        return system_prompt

    def get_chat_history(self, context, user_query):
        """
        Получает историю сообщений для чата
        """

        chat_id = context.chat_id
        if not chat_id:
            raise ValueError("chat_id is missing or empty")
        history = self.message_repo.get_by_chat(chat_id, limit=50, include_archived=False, reverse=True)
        cleaned_history = self.clean_history(history, user_query)
        return cleaned_history

    @abstractmethod
    async def handle(self, user_query: str, context: SimpleNamespace, tools: list[BaseTool],
                     stage_callback=None) -> str:
        pass
