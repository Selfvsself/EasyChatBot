from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


class LLMClient:

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.client = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            timeout=60.0,
            max_retries=2,
        )

    @staticmethod
    def _to_langchain_messages(messages: list[dict]):
        converted = []
        for message in messages:
            role = message.get("role", "user")
            content = str(message.get("content", ""))

            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))

        return converted

    async def chat(self, messages: list[dict]) -> str:
        lc_messages = self._to_langchain_messages(messages)
        response = await self.client.ainvoke(lc_messages)
        return response.content if isinstance(response.content, str) else str(response.content)
