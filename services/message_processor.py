from handlers.default_handler import DefaultHandler
from handlers.translator_handler import TranslatorHandler


class MessageProcessor:

    def __init__(self, message_repo, chat_repo, app_repo, llm_client):
        self.message_repo = message_repo
        self.chat_repo = chat_repo
        self.app_repo = app_repo
        self.llm = llm_client
        self.handlers = {
            "english-translator": TranslatorHandler(llm_client, message_repo),
            "russian-translator": TranslatorHandler(llm_client, message_repo),
            "default": DefaultHandler(llm_client, message_repo)
        }

    async def process(self, chat_id: str, user_id: str, text) -> str:
        chat = self.chat_repo.get_by_id(chat_id)

        app = self.app_repo.get_by_id(chat.app_id)

        handler = self.handlers.get(app.code, self.handlers["default"])
        return await handler.handle(chat, app, text)
