from __future__ import annotations

import json
import logging
import re
from typing import Iterable

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, ValidationError


class AgentOrchestrator:
    class ValidationDecision(BaseModel):
        is_complete: bool = True
        is_correct: bool = True
        needs_retry: bool = False
        feedback: str | None = None

    def __init__(self, llm_client, max_iterations: int = 3):
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.max_validation_retries = 2

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
    def _extract_user_query(messages) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage) and isinstance(message.content, str):
                return message.content
        return ""

    @staticmethod
    def _is_tool_call_parse_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "error parsing tool call" in text or "invalid character" in text

    async def _chat_without_tools(self, base_messages) -> str:
        guided_messages = list(base_messages) + [
            HumanMessage(
                content=(
                    "Tools are unavailable in this chat. "
                    "Do not output tool calls"
                )
            )
        ]
        return await self.llm_client.chat(guided_messages)

    async def _invoke_agent_once(self, graph, messages, stage_callback=None) -> tuple[str, list]:
        result = await graph.ainvoke(
            {"messages": messages},
            config={
                # In ReAct flow each tool step involves multiple graph nodes.
                "recursion_limit": self.max_iterations * 2 + 2
            },
        )
        await self._emit_stage(stage_callback, stage="typing")
        final = self._extract_final_text(result.get("messages", []))
        return final or "I could not produce a final answer.", result.get("messages", [])

    async def _validate_answer(
        self,
        user_query: str,
        answer: str,
    ) -> ValidationDecision:
        if not user_query.strip():
            return self.ValidationDecision(is_complete=True, is_correct=True, needs_retry=False, feedback=None)

        validation_prompt = [
            {
                "role": "system",
                "content": (
                    "Validate assistant answer quality. Return JSON only with keys: "
                    "is_complete, is_correct, needs_retry, feedback. "
                    "Set needs_retry=true if answer is incomplete/incorrect or lacks grounding."
                ),
            },
            {
                "role": "user",
                "content": f"User query:\n{user_query}\n\nAssistant answer:\n{answer}",
            },
        ]
        raw = await self.llm_client.chat(validation_prompt)
        try:
            cleaned = re.sub(r"```(?:json)?\n?|```", "", (raw or "")).strip()
            payload = json.loads(cleaned)
            return self.ValidationDecision.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError):
            return self.ValidationDecision(is_complete=True, is_correct=True, needs_retry=False, feedback=None)

    async def run(self, base_messages, chat, app, tools, stage_callback=None) -> str:
        await self._emit_stage(stage_callback, stage="thinking")

        lc_tools = self._build_tools(chat=chat, app=app, tools=tools, stage_callback=stage_callback)
        if not lc_tools:
            await self._emit_stage(stage_callback, stage="typing")
            response = await self._chat_without_tools(base_messages)
            return response

        graph = create_react_agent(
            model=self.llm_client.client,
            tools=lc_tools,
        )

        try:
            current_messages = list(base_messages)
            user_query = self._extract_user_query(base_messages)

            for attempt in range(1, self.max_validation_retries + 2):
                final, result_messages = await self._invoke_agent_once(
                    graph=graph,
                    messages=current_messages,
                    stage_callback=stage_callback,
                )

                if attempt >= self.max_validation_retries + 1:
                    return final

                await self._emit_stage(
                    stage_callback,
                    stage="agent_validating",
                    metadata={"attempt": attempt},
                )
                verdict = await self._validate_answer(user_query=user_query, answer=final)
                if not verdict.needs_retry and verdict.is_complete and verdict.is_correct:
                    return final

                feedback = (verdict.feedback or "Answer seems incomplete or uncertain.").strip()
                await self._emit_stage(
                    stage_callback,
                    stage="agent_replanning",
                    metadata={"attempt": attempt, "feedback": feedback},
                )
                current_messages = list(result_messages) + [
                    HumanMessage(
                        content=(
                            "Please revise your previous answer. "
                            f"Validation feedback: {feedback} "
                            "You may call different tools if needed."
                        )
                    )
                ]

            return final
        except Exception as exc:
            if self._is_tool_call_parse_error(exc):
                logging.warning(
                    "Tool-call parse error in agent runtime. Falling back to plain model response: %s",
                    exc,
                )
                await self._emit_stage(stage_callback, stage="typing")
                return await self._chat_without_tools(base_messages)
            raise
