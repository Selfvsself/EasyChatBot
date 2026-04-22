from .base_handler import BaseHandler
from services.agent_orchestrator import AgentOrchestrator


class DefaultHandler(BaseHandler):
    MAX_ITERATIONS = 3

    def __init__(self, llm_client, message_repo):
        super().__init__(llm_client, message_repo)
        self.orchestrator = AgentOrchestrator(llm_client=llm_client, max_iterations=self.MAX_ITERATIONS)

    async def handle(self, chat, app, text, tools, stage_callback=None):
        extracted_facts = "None"
        if chat.memory_summary:
            extracted_facts = f"\n\nChat memory (summary of archived messages):\n{chat.memory_summary.strip()}"
        history = self.message_repo.get_by_chat(chat.id, limit=50, include_archived=False)
        system_prompt = app.system_prompt.replace("[INSERT_PREVIOUS_FACTS_HERE]", extracted_facts)

        prompt = self.build_prompt(system=system_prompt, history=list(reversed(history)), text=text)
        return await self.orchestrator.run(
            user_query=text,
            history=prompt,
            context=extracted_facts,
            tools=tools,
            stage_callback=stage_callback,
        )
