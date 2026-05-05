import asyncio
import json
import logging

from handlers.pipelines.pipeline_factory import PipelineFactory
from handlers.pipelines.pipeline_orchestrator import PipelineOrchestrator
from handlers.pipelines.steps.step_context import StepContext
from services.llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

system_prompt = """
You are a helpful and knowledgeable AI assistant with real-time web search capabilities. 

Your goal is to provide direct, fluid, and natural answers to user queries based on the most up-to-date information available.

User Context (Reference only):
The following information provides context about the user's ongoing work to ensure translation accuracy:
None

CRITICAL INSTRUCTIONS: 
- LANGUAGE ADAPTIVITY: Always respond in the same language the user used for their last message. If the user asks in Russian, answer in Russian; if in English, answer in English.
- NO CITATIONS: DO NOT provide any links, URLs, or citations in your response.
- NO SOURCE MENTIONING: DO NOT mention the names of specific websites or sources from which you gathered information.
- CONSOLIDATED KNOWLEDGE: Provide the information as your own knowledge in a conversational tone.
- SYNTHESIS: Synthesize search results into a coherent answer without using phrases like "According to [Website]" or "Source: [Link]".

Operational Guidelines:
1. Search the web when the query requires current events, specific data points, or topics beyond your training data.
2. Structure your response clearly using logical paragraphs or bullet points where appropriate.
3. Be concise and prioritize the most relevant information.
4. If a query is ambiguous, ask for clarification before searching.
5. Accuracy First: Do not hallucinate. If you cannot find the answer via search, state it clearly.
"""


async def emit_stage(stage: str, metadata: dict | None = None):
    payload = {
        "event": "agent_stage",
        "stage": stage,
        "metadata": metadata or {},
    }
    print(json.dumps(payload))


async def run(user_query: str, stage_callback=None):
    llm_client = LLMClient(
        base_url="http://192.168.1.73:11434",
        model="gpt-32k"
    )

    factory = PipelineFactory(llm_client=llm_client)
    orchestrator = PipelineOrchestrator()

    context = StepContext(
        user_input=user_query,
        # history=[{
        #     "role": "user",
        #     "content": "а какие книги выйдут в этом месяце"},
        #     {
        #         "role": "assistant",
        #         "content": "В мае 2026 года в русскоязычной книжной индустрии появятся следующие новинки:\n\n- **«История дворца Куньнин» (первый том)** – роман, впервые публикуется именно в этом месяце.  \n- **«Загадка этажа номер 12»** – новелла Нина Ханъи, дата выпуска была перенесена, и теперь она выходит в мае.  \n- **«Individuum»** – новая книга, релиз запланирован на май 2026. Информация о авторе и издателе пока не уточнена.\n\nЕсли понадобится более подробная информация о каждом из этих изданий, дайте знать!"
        #     }],
        history=[],
        system_prompt=system_prompt
    )

    pipeline = factory.web_search_agent()
    return await orchestrator.run(pipeline, context, stage_callback)


if __name__ == "__main__":
    # result = asyncio.run(run("Напиши код на python который отсчитывает таймер", emit_stage))
    # result = asyncio.run(run("Найди какие пет проекты можно сделать для себя имея llm 20b", emit_stage))
    result = asyncio.run(run("Какие фильмы выходят в мае", emit_stage))
    # result = asyncio.run(run("сколько стоит хавал в москве", emit_stage))
    # result = asyncio.run(run("какое сейчас число", emit_stage))
    # result = asyncio.run(run("а фильмы", emit_stage))
    # result = asyncio.run(run("а куда мы", emit_stage))
    # result = asyncio.run(run("расскажи о себе", emit_stage))
    # result = asyncio.run(run("а в июне", emit_stage))
    print(result)
