from .base_handler import BaseHandler
from services.agent_orchestrator import AgentOrchestrator


class AgentHandler(BaseHandler):
    MAX_ITERATIONS = 3

    def __init__(self, llm_client, message_repo):
        super().__init__(llm_client, message_repo)
        self.orchestrator = AgentOrchestrator(llm_client=llm_client, max_iterations=self.MAX_ITERATIONS)

    async def handle(self, chat, app, text, tools):
        history = self.message_repo.get_by_chat(chat.id, limit=20)
        system = (
            f"{app.system_prompt}\n\n"
            "You can use available tools when needed. "
            "Use tools only for factual retrieval, then provide a direct final answer."
        )
        prompt = self.build_prompt(system=system, history=list(reversed(history)), text=text)
        return await self.orchestrator.run(base_messages=prompt, chat=chat, app=app, tools=tools)
