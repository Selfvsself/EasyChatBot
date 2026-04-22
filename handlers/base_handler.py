from abc import abstractmethod

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class BaseHandler:

    def __init__(self, llm_client, message_repo):
        self.llm = llm_client
        self.message_repo = message_repo

    async def processing_tools(self, tools, chat, app, text):
        tool_result = []
        for tool in tools:
            response = await tool.processing(chat, app, text)
            tool_result.append(response)
        return "\n".join(tool_result)

    @staticmethod
    def _to_langchain_history(history):
        converted = []
        for message in history:
            role = getattr(message, "role", "user")
            content = str(getattr(message, "text", ""))

            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))

        return converted

    def build_prompt(self, system, history, text):
        history_messages = self._to_langchain_history(history)

        has_duplicate_user_tail = (
            bool(history_messages)
            and isinstance(history_messages[-1], HumanMessage)
            and history_messages[-1].content == text
        )

        if has_duplicate_user_tail:
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", "{system}"),
                    MessagesPlaceholder(variable_name="history"),
                ]
            )
            return prompt_template.format_messages(
                system=system,
                history=history_messages,
            )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "{system}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{text}"),
            ]
        )
        return prompt_template.format_messages(
            system=system,
            history=history_messages,
            text=text,
        )

    @abstractmethod
    async def handle(self, chat, app, text, tools, stage_callback=None):
        pass
