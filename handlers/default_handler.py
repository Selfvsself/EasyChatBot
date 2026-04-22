from .base_handler import BaseHandler


class DefaultHandler(BaseHandler):

    async def handle(self, chat, app, text, tools):
        tools_result = await self.processing_tools(tools, chat, app, text)
        system_prompt = app.system_prompt + "\n" + tools_result

        history = self.message_repo.get_by_chat(chat.id, limit=20)
        prompt = self.build_prompt(system_prompt, list(reversed(history)), text=text)
        return await self.llm.chat(prompt)
