from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama


class ResponseMode(str, Enum):
    TEXT = "text"
    JSON = "json"


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
        temperature: float = 0.85,
        top_p: float = 0.92,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        repeat_last_n: int = 128,
        num_predict: int = 512,
    ):
        self.base_url = base_url
        self.model = model

        self._options = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
            "num_predict": num_predict,
        }

        self._client = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
            options=self._options,
        )

    @property
    def client(self) -> ChatOllama:
        return self._client

    async def chat(
        self,
        messages: list[dict | BaseMessage],
        *,
        mode: ResponseMode = ResponseMode.TEXT,
        tools: Sequence[BaseTool] | None = None,
        tool_choice: str | None = None,
        return_message: bool = False,
        **kwargs: Any,
    ) -> str | AIMessage:

        lc_messages = self._to_langchain_messages(messages)

        llm = self._client

        if tools:
            llm = llm.bind_tools(list(tools), tool_choice=tool_choice)

        invoke_kwargs: dict[str, Any] = {}

        if mode == ResponseMode.JSON:
            if tools:
                raise ValueError(
                    "JSON mode with tools is ambiguous. "
                    "Use tools-only flow or disable tools."
                )
            invoke_kwargs["format"] = "json"

        invoke_kwargs.update(kwargs)

        response = await llm.ainvoke(lc_messages, **invoke_kwargs)

        if return_message:
            return response

        content = response.content
        return content if isinstance(content, str) else str(content)

    async def chat_text(
        self,
        messages: list[dict | BaseMessage],
        **kwargs: Any,
    ) -> str:
        return await self.chat(messages, mode=ResponseMode.TEXT, **kwargs)

    async def chat_json(
        self,
        messages: list[dict | BaseMessage],
        **kwargs: Any,
    ) -> str:
        return await self.chat(messages, mode=ResponseMode.JSON, **kwargs)

    async def chat_with_tools(
        self,
        messages: list[dict | BaseMessage],
        tools: Sequence[BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> str:
        return await self.chat(
            messages,
            mode=ResponseMode.TEXT,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    @staticmethod
    def _to_langchain_messages(
        messages: Iterable[dict | BaseMessage],
    ) -> list[BaseMessage]:
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