from abc import abstractmethod


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

    def build_prompt(self, system, history, text):
        messages = [{"role": "system", "content": system}]

        for m in history:
            messages.append({"role": m.role, "content": m.text})

        is_duplicate = len(messages) > 1 and messages[-1]["content"] == text

        if not is_duplicate:
            messages.append({
                "role": "user",
                "content": text
            })

        return messages

    @abstractmethod
    async def handle(self, chat, app, text, tools):
        pass
