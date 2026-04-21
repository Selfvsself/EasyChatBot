from .base_handler import BaseHandler


class DefaultHandler(BaseHandler):

    async def handle(self, chat, app, text):
        history = self.message_repo.get_by_chat(chat.id, limit=20)
        prompt = self.build_prompt(app.system_prompt, list(reversed(history)), text=text)
        return await self.llm.chat(prompt)
