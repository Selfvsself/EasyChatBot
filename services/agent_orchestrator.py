from __future__ import annotations

import json
import logging
import re
from typing import Iterable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ValidationError

from services.agent_factory import AgentFactory


class AgentOrchestrator:
    class ValidationDecision(BaseModel):
        is_sufficient: bool = True
        critique_score: int = 0
        detailed_critique: str | None = None
        missing_info_points: list[str] | None = None
        action_recommendation: str | None = None

    def __init__(self, llm_client, max_iterations: int = 3):
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        # Keep validation lightweight to reduce latency and token usage.
        self.max_validation_retries = 1
        self.agent_factory = AgentFactory(llm_client)

    @staticmethod
    async def _emit_stage(stage_callback, stage: str, metadata: dict | None = None):
        if stage_callback:
            await stage_callback(stage=stage, metadata=metadata or {})

    @staticmethod
    def _build_tools(history, context, tools: Iterable, stage_callback=None) -> list[StructuredTool]:
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
                        history=history,
                        context=context,
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

    @staticmethod
    def _message_role(message) -> str:
        if isinstance(message, SystemMessage):
            return "system"
        if isinstance(message, AIMessage):
            return "assistant"
        if isinstance(message, HumanMessage):
            return "user"
        return "user"

    @classmethod
    def _build_validation_context(cls, messages, max_messages: int = 12) -> str:
        if not messages:
            return "(empty)"

        compact = []
        for message in messages[-max_messages:]:
            role = cls._message_role(message)
            if role == "system":
                continue
            content = str(getattr(message, "content", "")).strip()
            if not content:
                continue
            compact.append(f"{role}: {content}")

        return "\n".join(compact) if compact else "(empty)"

    async def _chat_without_tools(self, base_messages) -> str:
        guided_messages = list(base_messages) + [
            HumanMessage(
                content=(
                    "Tools are unavailable in this chat. "
                    "Answer directly in plain natural language. "
                    "Do not output tool calls, function stubs, or JSON arrays."
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

    @staticmethod
    def _build_messages_block(messages):
        lines = []
        for message in messages:
            role = getattr(message, "type", "user")
            if role == "system":
                continue
            text = str(getattr(message, "text", "")).strip().replace("\n", " ")
            if role == "ai":
                role = "assistant"
            else:
                role = "user"
            if text:
                lines.append(f"{role}: {text}")
        return "\n".join(lines)

    async def _validate_answer(
            self,
            user_query: str,
            answer: str,
            history: list[str],
            context: str
    ) -> ValidationDecision:
        if not user_query.strip():
            return self.ValidationDecision(is_sufficient=True, critique_score=10, detailed_critique=None, missing_info_points=None, action_recommendation=None)

        formatted_history = self._build_messages_block(history)

        validation_prompt = [
            {
                "role": "system",
                "content": (
                    "Role: "
                    "You are a rigorous Quality Assurance Agent specializing in information accuracy and completeness. "
                    "Your goal is to evaluate whether the gathered search results and the proposed response fully "
                    "satisfy the user's intent based on the available context. "
                    "Input Data: "
                    "1. User Request: The original question or task. "
                    "2. Search Results Summary: The raw or summarized data retrieved from the web. "
                    "3. User Context (Facts): Long-term established facts about the user. "
                    "4. Message History: The flow of the current conversation. "
                    "Evaluation Tasks: "
                    "1. Completeness: Does the information cover ALL parts of the user's request? "
                    "2. Accuracy: Are there any contradictions between the search results and the established \"User Context\" or \"Message History\"? "
                    "3. Relevance: Is the information up-to-date and specific to the user's current situation? "
                    "4. Gap Identification: What specific details are missing to provide a perfect answer? "
                    "Output Format: "
                    "Return a valid JSON object. "
                    "JSON Schema: "
                    "{ "
                    "\"detailed_critique\": \"Explanation of what is missing or what contradicts the history/facts\", "
                    "\"missing_info_points\": [\"Point 1\", \"Point 2\"], "
                    "\"is_sufficient\": boolean, "
                    "\"critique_score\": integer (1-10, where 10 is perfectly sufficient), "
                    "\"action_recommendation\": \"Pass to final response | Perform additional search | Ask user for clarification\" "
                    "} "
                    "Constraints: "
                    "- Be hyper-critical. It is better to perform an extra search than to provide a shallow or partially incorrect answer. "
                    "- If the search results provide generic info but the User Context requires something specific, mark 'is_sufficient' as false."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User Request:\n{user_query}\n\n"
                    f"Search Results Summary:\n{answer}\n\n"
                    f"User Context (Facts): {context}\n\n"
                    f"Message History:\n{formatted_history}"
                ),
            },
        ]
        raw = await self.llm_client.chat(validation_prompt, response_format="json")
        try:
            cleaned = re.sub(r"```(?:json)?\n?|```", "", (raw or "")).strip()
            payload = json.loads(cleaned)
            return self.ValidationDecision.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError):
            return self.ValidationDecision(is_sufficient=True, critique_score=10, detailed_critique=None, missing_info_points=None, action_recommendation=None)

    async def run(self, user_query, history, context, tools, stage_callback=None,
                  response_format: str | None = None) -> str:
        await self._emit_stage(stage_callback, stage="thinking")

        lc_tools = self._build_tools(history=history, context=context, tools=tools, stage_callback=stage_callback)
        if not lc_tools:
            await self._emit_stage(stage_callback, stage="typing")
            return await self.llm_client.chat(history, response_format=response_format)

        graph = self.agent_factory.create_react(lc_tools)

        try:
            current_messages = list(history)

            final = None

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
                verdict = await self._validate_answer(
                    user_query=user_query,
                    answer=final,
                    history=history,
                    context=context
                )

                # If user did not request sources, do not reject only because citations/links are missing.
                if verdict.is_sufficient and verdict.critique_score > 6:
                    return final

                if verdict.critique_score > 8:
                    return final

                feedback = (verdict.detailed_critique or "Answer seems incomplete or uncertain.").strip()
                feedbacks = ",".join(verdict.missing_info_points)
                action_recommendation = verdict.action_recommendation
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
                            f"Missing info points: {feedbacks} "
                            f"Action recommendation: {action_recommendation} "
                            "You may call different tools if needed."
                        )
                    )
                ]

            if final:
                return final
            else:
                await self._emit_stage(stage_callback, stage="typing")
                return await self.llm_client.chat(history, response_format=response_format)
        except Exception as exc:
            if self._is_tool_call_parse_error(exc):
                logging.warning(
                    "Tool-call parse error in agent runtime. Falling back to plain model response: %s",
                    exc,
                )
                await self._emit_stage(stage_callback, stage="typing")
                return await self.llm_client.chat(history, response_format=response_format)
            raise
