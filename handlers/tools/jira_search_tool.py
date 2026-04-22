from __future__ import annotations

import json
import re
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from core.config import settings
from .base_tool import BaseTool


class JiraSearchTool(BaseTool):
    agent_tool_name = "jira_search"
    agent_tool_description = (
        "Search Jira issues by natural language or explicit JQL (prefix with JQL:). "
        "The tool can inspect issue descriptions/comments and refine queries automatically."
    )

    MAX_RESULTS = 5
    HTTP_TIMEOUT = 15.0
    MAX_REFINE_ATTEMPTS = 3

    class QueryPlan(BaseModel):
        jql: str
        rationale: str | None = None

    class AnalysisResult(BaseModel):
        answer: str
        is_complete: bool = True
        is_correct: bool = True
        needs_retry: bool = False
        retry_query: str | None = None
        evidence: list[str] = Field(default_factory=list)

    def __init__(self, llm_client, message_repo):
        super().__init__(llm_client, message_repo)
        self.base_url = (settings.JIRA_BASE_URL or "").rstrip("/")
        self.email = settings.JIRA_EMAIL or ""
        self.api_token = settings.JIRA_API_TOKEN or ""
        self.pat = settings.JIRA_PAT or ""
        self.project_key = settings.JIRA_PROJECT_KEY or ""

    def _is_configured(self) -> bool:
        has_auth = bool(self.pat) or (bool(self.email) and bool(self.api_token))
        return bool(self.base_url) and has_auth

    def _build_jql(self, query: str) -> str:
        clean_query = (query or "").strip()
        if clean_query.upper().startswith("JQL:"):
            return clean_query[4:].strip()

        escaped = clean_query.replace('"', '\\"')
        base_filter = f'project = "{self.project_key}" AND ' if self.project_key else ""
        return f'{base_filter}text ~ "{escaped}" ORDER BY updated DESC'

    def _auth_headers(self) -> tuple[dict[str, str], tuple[str, str] | None]:
        headers = {"Accept": "application/json"}
        auth: tuple[str, str] | None = None

        if self.pat:
            headers["Authorization"] = f"Bearer {self.pat}"
        else:
            auth = (self.email, self.api_token)

        return headers, auth

    @staticmethod
    def _strip_fences(text: str) -> str:
        return re.sub(r"```(?:json)?\n?|```", "", text or "").strip()

    async def _plan_jql_with_llm(self, user_query: str) -> str:
        plan_system = (
            "You generate JQL for Jira search. "
            "Return JSON only with keys: jql, rationale."
            "RULES:"
            "1. Always use the '~' operator with the 'text' field for the user's input (e.g., text ~ \"value\")."
            "2. Do not attempt to guess field names like 'status' or 'summary' unless specifically mentioned."
            "3. Always append 'ORDER BY updated DESC' to the JQL."
            "4. If a project context is provided, include it, otherwise focus on the text search."
            "Example:"
            "Input: \"Invite for submission\""
            "Output: {\"jql\": \"text ~ \"Invite for submission\" ORDER BY updated DESC\", \"rationale\": \"Performing a full-text search for the phrase.\"}"
        )
        prompt = self.build_prompt(
            system=plan_system,
            history=[],
            text=(
                f"User query:\n{user_query}\n\n"
                f"Project key: {self.project_key or 'not set'}\n"
                "Return JSON only."
            ),
        )

        raw = await self.llm.chat(prompt)
        try:
            payload = json.loads(self._strip_fences(raw))
            model = self.QueryPlan.model_validate(payload)
            return model.jql.strip()
        except (json.JSONDecodeError, ValidationError, TypeError):
            return self._build_jql(user_query)

    async def _search_by_jql(self, jql: str) -> tuple[list[dict[str, str]], str | None]:
        if not self._is_configured():
            return [], (
                "Jira search is not configured. "
                "Set JIRA_BASE_URL and auth (JIRA_PAT or JIRA_EMAIL + JIRA_API_TOKEN)."
            )

        headers, auth = self._auth_headers()
        params = {
            "jql": jql,
            "maxResults": self.MAX_RESULTS,
            "fields": "summary,status,assignee,priority,updated",
        }

        endpoints = [
            f"{self.base_url}/rest/api/3/search/jql",
            f"{self.base_url}/rest/api/3/search",
        ]

        last_error: str | None = None
        async with httpx.AsyncClient(timeout=self.HTTP_TIMEOUT) as client:
            for endpoint in endpoints:
                try:
                    response = await client.get(endpoint, params=params, headers=headers, auth=auth)
                    response.raise_for_status()
                    payload = response.json()
                    return self._extract(payload), None
                except Exception as exc:
                    last_error = str(exc)

        return [], f"Jira search failed: {last_error or 'unknown error'}"

    async def _search(self, query: str) -> tuple[list[dict[str, str]], str | None]:
        return await self._search_by_jql(self._build_jql(query))

    @staticmethod
    def _adf_to_text(node: Any) -> str:
        parts: list[str] = []

        def walk(obj: Any):
            if isinstance(obj, dict):
                txt = obj.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
                content = obj.get("content")
                if isinstance(content, list):
                    for item in content:
                        walk(item)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(node)
        return re.sub(r"\s+", " ", " ".join(parts)).strip()

    async def _fetch_issue_details(
        self,
        issue_key: str,
        stage_callback=None,
    ) -> dict[str, Any] | None:
        if stage_callback:
            await stage_callback(
                stage="jira_ticket_opening",
                metadata={"key": issue_key, "url": f"{self.base_url}/browse/{issue_key}"},
            )

        headers, auth = self._auth_headers()
        endpoint = f"{self.base_url}/rest/api/3/issue/{issue_key}"
        params = {
            "fields": "summary,status,assignee,priority,updated,description,comment",
        }

        try:
            async with httpx.AsyncClient(timeout=self.HTTP_TIMEOUT) as client:
                response = await client.get(endpoint, params=params, headers=headers, auth=auth)
                response.raise_for_status()
                issue = response.json()
        except Exception:
            return None

        fields = issue.get("fields") or {}
        description_text = self._adf_to_text(fields.get("description"))
        comments = ((fields.get("comment") or {}).get("comments")) or []
        comment_texts = []
        for c in comments[:5]:
            body_text = self._adf_to_text(c.get("body"))
            if body_text:
                comment_texts.append(body_text)

        return {
            "key": issue_key,
            "summary": fields.get("summary") or "",
            "status": ((fields.get("status") or {}).get("name")) or "",
            "priority": ((fields.get("priority") or {}).get("name")) or "n/a",
            "assignee": ((fields.get("assignee") or {}).get("displayName")) or "Unassigned",
            "updated": fields.get("updated") or "",
            "url": f"{self.base_url}/browse/{issue_key}",
            "description": description_text,
            "comments": comment_texts,
        }

    async def _fetch_many_issue_details(self, issues: list[dict[str, str]], stage_callback=None) -> list[dict[str, Any]]:
        detailed: list[dict[str, Any]] = []
        for item in issues:
            key = item.get("key")
            if not key:
                continue
            data = await self._fetch_issue_details(key, stage_callback=stage_callback)
            if data:
                detailed.append(data)
        return detailed

    async def _analyze_with_llm(
        self,
        user_query: str,
        jql: str,
        issues: list[dict[str, Any]],
    ) -> AnalysisResult:
        if not issues:
            return self.AnalysisResult(
                answer="No relevant Jira ticket details were found.",
                is_complete=False,
                is_correct=True,
                needs_retry=True,
                retry_query=f"{user_query} blocker root cause",
                evidence=[],
            )

        blocks = []
        for idx, issue in enumerate(issues, start=1):
            comment_block = "\n".join(f"- {c}" for c in issue["comments"]) if issue["comments"] else "- no comments"
            blocks.append(
                f"[Issue {idx}] {issue['key']} - {issue['summary']}\n"
                f"Status: {issue['status']}; Priority: {issue['priority']}; Assignee: {issue['assignee']}\n"
                f"Updated: {issue['updated']}\n"
                f"URL: {issue['url']}\n"
                f"Description: {issue['description'] or 'n/a'}\n"
                f"Comments:\n{comment_block}"
            )

        analysis_system = (
            "You are a Jira analyst. "
            "Return JSON only with keys: answer, is_complete, is_correct, needs_retry, retry_query, evidence. "
            "evidence must be a list of ticket keys or short facts. "
            "Set needs_retry=true only if data is insufficient or contradictory."
        )
        analysis_prompt = self.build_prompt(
            system=analysis_system,
            history=[],
            text=(
                f"User query:\n{user_query}\n\n"
                f"JQL used:\n{jql}\n\n"
                "Issue details:\n\n"
                + "\n\n".join(blocks)
            ),
        )

        raw = await self.llm.chat(analysis_prompt)
        try:
            payload = json.loads(self._strip_fences(raw))
            return self.AnalysisResult.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError):
            return self.AnalysisResult(
                answer=raw.strip() or "Could not produce a Jira analysis answer.",
                is_complete=True,
                is_correct=True,
                needs_retry=False,
                retry_query=None,
                evidence=[],
            )

    async def _best_effort_answer_with_llm(
        self,
        user_query: str,
        jql: str,
        issues: list[dict[str, Any]],
    ) -> str:
        if not issues:
            return "No relevant Jira ticket details were found."

        blocks = []
        for idx, issue in enumerate(issues, start=1):
            comment_block = "\n".join(f"- {c}" for c in issue["comments"]) if issue["comments"] else "- no comments"
            blocks.append(
                f"[Issue {idx}] {issue['key']} - {issue['summary']}\n"
                f"Status: {issue['status']}; Priority: {issue['priority']}; Assignee: {issue['assignee']}\n"
                f"Updated: {issue['updated']}\n"
                f"URL: {issue['url']}\n"
                f"Description: {issue['description'] or 'n/a'}\n"
                f"Comments:\n{comment_block}"
            )

        prompt = self.build_prompt(
            system=(
                "You are a Jira analyst. Provide the best possible answer from available ticket data. "
                "If data is partial, explicitly mention uncertainty. Return plain text only."
            ),
            history=[],
            text=(
                f"User query:\n{user_query}\n\n"
                f"JQL used:\n{jql}\n\n"
                "Issue details:\n\n"
                + "\n\n".join(blocks)
            ),
        )
        answer = await self.llm.chat(prompt)
        return (answer or "").strip() or "Could not produce a Jira answer."

    def _extract(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        issues = payload.get("issues") or []
        results: list[dict[str, str]] = []

        for issue in issues:
            key = issue.get("key")
            fields = issue.get("fields") or {}
            summary = fields.get("summary") or ""
            status = ((fields.get("status") or {}).get("name")) or ""
            assignee = ((fields.get("assignee") or {}).get("displayName")) or "Unassigned"
            priority = ((fields.get("priority") or {}).get("name")) or "n/a"
            updated = fields.get("updated") or ""
            url = f"{self.base_url}/browse/{key}" if key else ""

            if not key:
                continue

            results.append(
                {
                    "key": key,
                    "summary": summary,
                    "status": status,
                    "assignee": assignee,
                    "priority": priority,
                    "updated": updated,
                    "url": url,
                }
            )

        return results[:self.MAX_RESULTS]

    def _format_results(self, results: list[dict[str, str]]) -> str:
        if not results:
            return "No Jira tickets found."

        lines = []
        for idx, item in enumerate(results, start=1):
            lines.append(
                f"{idx}. {item['key']} - {item['summary']}\n"
                f"Status: {item['status']}; Priority: {item['priority']}; Assignee: {item['assignee']}\n"
                f"Updated: {item['updated']}\n"
                f"URL: {item['url']}"
            )
        return "\n\n".join(lines)

    async def processing(self, chat, app, text):
        results, error = await self._search(text)
        if error:
            return f"Jira search tool error: {error}"

        return (
            "Jira ticket search results are provided below. "
            "Use ticket status and priority as factual context. "
            "If no relevant ticket is present, say so clearly.\n\n"
            f"{self._format_results(results)}"
        )

    async def run_for_agent(self, query: str, chat=None, app=None, stage_callback=None) -> str:
        working_query = query
        last_analysis: JiraSearchTool.AnalysisResult | None = None
        last_jql = self._build_jql(query)
        last_detailed: list[dict[str, Any]] = []

        for attempt in range(1, self.MAX_REFINE_ATTEMPTS + 1):
            is_last_attempt = attempt == self.MAX_REFINE_ATTEMPTS

            if stage_callback:
                await stage_callback(
                    stage="jira_query_planning",
                    metadata={"attempt": attempt, "query": working_query},
                )

            jql = await self._plan_jql_with_llm(working_query)
            last_jql = jql or self._build_jql(working_query)

            if stage_callback:
                await stage_callback(
                    stage="jira_query_running",
                    metadata={"attempt": attempt, "jql": last_jql},
                )

            issues, error = await self._search_by_jql(last_jql)
            if error:
                return f"Jira search tool error: {error}"

            detailed = await self._fetch_many_issue_details(issues, stage_callback=stage_callback)
            last_detailed = detailed

            if stage_callback and detailed:
                await stage_callback(
                    stage="sources_used",
                    metadata={
                        "sources": [
                            {
                                "title": f"{item['key']} - {item['summary']}",
                                "url": item["url"],
                            }
                            for item in detailed
                        ]
                    },
                )

            if is_last_attempt:
                best_effort = await self._best_effort_answer_with_llm(query, last_jql, detailed)
                ticket_line = ", ".join(i["key"] for i in detailed) if detailed else "none"
                return (
                    f"JQL used: {last_jql}\n"
                    f"Tickets inspected: {ticket_line}\n"
                    f"Evidence: n/a\n\n"
                    f"Answer:\n{best_effort}"
                )

            if stage_callback:
                await stage_callback(
                    stage="jira_validating",
                    metadata={"attempt": attempt, "tickets": [i["key"] for i in detailed]},
                )

            analysis = await self._analyze_with_llm(query, last_jql, detailed)
            last_analysis = analysis

            should_retry = (
                attempt < self.MAX_REFINE_ATTEMPTS
                and (analysis.needs_retry or not analysis.is_complete or not analysis.is_correct)
                and bool((analysis.retry_query or "").strip())
            )
            if should_retry:
                working_query = analysis.retry_query.strip()
                continue

            evidence_line = ", ".join(analysis.evidence) if analysis.evidence else "n/a"
            ticket_line = ", ".join(i["key"] for i in detailed) if detailed else "none"
            return (
                f"JQL used: {last_jql}\n"
                f"Tickets inspected: {ticket_line}\n"
                f"Evidence: {evidence_line}\n\n"
                f"Answer:\n{analysis.answer}"
            )

        if last_analysis:
            evidence_line = ", ".join(last_analysis.evidence) if last_analysis.evidence else "n/a"
            ticket_line = ", ".join(i["key"] for i in last_detailed) if last_detailed else "none"
            return (
                f"JQL used: {last_jql}\n"
                f"Tickets inspected: {ticket_line}\n"
                f"Evidence: {evidence_line}\n\n"
                f"Answer:\n{last_analysis.answer}"
            )

        return "Jira search completed but no answer could be produced."
