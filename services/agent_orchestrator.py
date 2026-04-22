from __future__ import annotations

import logging
from typing import Iterable

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent


class AgentOrchestrator:
    def __init__(self, llm_client, max_iterations: int = 3):
        self.llm_client = llm_client
        self.max_iterations = max_iterations

    @staticmethod
    async def _emit_stage(stage_callback, stage: str, metadata: dict | None = None):
        if stage_callback:
            await stage_callback(stage=stage, metadata=metadata or {})

    @staticmethod
    def _build_tools(chat, app, tools: Iterable, stage_callback=None) -> list[StructuredTool]:
        tool_defs: list[StructuredTool] = []

        for tool in tools:
            tool_name = tool.__class__.__name__

            if hasattr(tool, "run_for_agent"):
                async def _runner(query: str, _tool=tool):
                    await AgentOrchestrator._emit_stage(
                        stage_callback,
                        stage="tool_running",
                        metadata={
                            "tool": getattr(_tool, "agent_tool_name", _tool.__class__.__name__.lower()),
                            "query": query,
                        },
                    )
                    return await _tool.run_for_agent(
                        query,
                        chat=chat,
                        app=app,
                        stage_callback=stage_callback,
                    )

                description = (
                    f"Use this tool when you need {tool_name} data. "
                    "Input must be a single concise natural language query."
                )
                tool_defs.append(
                    StructuredTool.from_function(
                        coroutine=_runner,
                        name=getattr(tool, "agent_tool_name", tool_name.lower()),
                        description=getattr(tool, "agent_tool_description", description),
                    )
                )

        return tool_defs

    @staticmethod
    def _extract_final_text(messages) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content.strip():
                return message.content
        return ""

    @staticmethod
    def _is_tool_call_parse_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "error parsing tool call" in text or "invalid character" in text

    async def run(self, base_messages, chat, app, tools, stage_callback=None) -> str:
        await self._emit_stage(stage_callback, stage="thinking")

        lc_tools = self._build_tools(chat=chat, app=app, tools=tools, stage_callback=stage_callback)
        if not lc_tools:
            await self._emit_stage(stage_callback, stage="typing")
            response = await self.llm_client.chat(base_messages)
            return response

        graph = create_react_agent(
            model=self.llm_client.client,
            tools=lc_tools,
        )

        try:
            result = await graph.ainvoke(
                {"messages": base_messages},
                config={
                    # In ReAct flow each tool step involves multiple graph nodes.
                    "recursion_limit": self.max_iterations * 2 + 2
                },
            )
            await self._emit_stage(stage_callback, stage="typing")
            final = self._extract_final_text(result.get("messages", []))
            return final or "I could not produce a final answer."
        except Exception as exc:
            if self._is_tool_call_parse_error(exc):
                logging.warning(
                    "Tool-call parse error in agent runtime. Falling back to plain model response: %s",
                    exc,
                )
                await self._emit_stage(stage_callback, stage="typing")
                return await self.llm_client.chat(base_messages)
            raise
