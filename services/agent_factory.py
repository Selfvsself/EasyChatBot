from typing import Sequence

from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from services.llm_client import LLMClient


class AgentFactory:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def create_react(
            self,
            tools: Sequence[BaseTool],
    ):
        return create_react_agent(
            model=self.llm_client.get_model(),
            tools=list(tools),
        )
