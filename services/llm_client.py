from typing import Any, Iterable

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama


class LLMClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

        self._client = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            timeout=60.0,
            max_retries=2,
        )

    def get_model(self):
        return self._client

    async def chat(
            self,
            messages: list[dict | BaseMessage],
            *,
            response_format: str | None = None,
            **kwargs: Any,
    ) -> str:
        lc_messages = self._to_langchain_messages(messages)

        invoke_kwargs: dict[str, Any] = {}

        if response_format == "json":
            invoke_kwargs["format"] = "json"

        invoke_kwargs.update(kwargs)

        response = await self._client.ainvoke(lc_messages, **invoke_kwargs)

        content = response.content
        return content if isinstance(content, str) else str(content)

    async def chat_json(
            self,
            messages: list[dict | BaseMessage],
            **kwargs: Any,
    ) -> str:
        return await self.chat(messages, response_format="json", **kwargs)

    @staticmethod
    def _to_langchain_messages(messages: Iterable[dict | BaseMessage]):
        converted: list[BaseMessage] = []

        for message in messages:
            if isinstance(message, BaseMessage):
                converted.append(message)
                continue

            role = message.get("role", "user")
            content = str(message.get("content", ""))

            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))

        return converted
