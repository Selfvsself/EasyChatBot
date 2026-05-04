from .pipelines.pipeline_factory import PipelineFactory
from .pipelines.pipeline_orchestrator import PipelineOrchestrator
from .pipelines.steps.pipeline_context import PipelineContext
from .base_handler import BaseHandler


class WebSearchHandler(BaseHandler):
    MAX_ITERATIONS = 3

    def __init__(self, llm_client, message_repo):
        super().__init__(llm_client, message_repo)
        self.pipeline = PipelineFactory(llm_client=llm_client)
        self.orchestrator = PipelineOrchestrator()

    async def handle(self, user_query, context, tools, stage_callback=None):
        history = self.get_chat_history(context, user_query)
        chat_memory = self.prepare_chat_memory(context)
        system_prompt = self.prepare_system_prompt(context, chat_memory)

        context = PipelineContext(
            user_input=user_query,
            history=history,
            system_prompt=system_prompt
        )
        pipeline = self.pipeline.web_search_pipeline()
        result = await self.orchestrator.run(pipeline, context, stage_callback)
        return result
