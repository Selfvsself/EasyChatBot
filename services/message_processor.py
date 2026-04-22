from types import SimpleNamespace

from handlers.default_handler import DefaultHandler
from handlers.translator_handler import TranslatorHandler

from handlers.tools.jira_search_tool import JiraSearchTool
from handlers.tools.web_search_tool import WebSearchTool


class MessageProcessor:

    def __init__(self, message_repo, chat_repo, app_repo, llm_client, app_tool_repo):
        self.message_repo = message_repo
        self.chat_repo = chat_repo
        self.app_repo = app_repo
        self.app_tool_repo = app_tool_repo
        self.llm = llm_client
        self.handlers = {
            "english-translator": TranslatorHandler(llm_client, message_repo),
            "russian-translator": TranslatorHandler(llm_client, message_repo),
            "duckduckgo-search": DefaultHandler(llm_client, message_repo),
            "default": DefaultHandler(llm_client, message_repo)
        }
        self.tools = {
            "web-search": WebSearchTool(llm_client, message_repo),
            "jira-search": JiraSearchTool(llm_client, message_repo)
        }

    async def process(self, chat_id: str, user_id: str, text) -> str:
        chat = self.chat_repo.get_by_id(chat_id)

        app = self.app_repo.get_by_id(chat.app_id)
        tool_codes = self.app_tool_repo.get_enabled_tool_codes_by_app(chat.app_id)
        tools = [self.tools[t_code] for t_code in tool_codes if t_code in self.tools]


        app_ctx = SimpleNamespace(
            system_prompt=app.system_prompt,
            tool_codes=tool_codes
        )

        handler = self.handlers.get(app.code, self.handlers["default"])
        return await handler.handle(chat, app_ctx, text, tools)
