from typing import Any

import httpx

from core.config import settings
from .base_tool import BaseTool


class JiraSearchTool(BaseTool):
    agent_tool_name = "jira_search"
    agent_tool_description = (
        "Search Jira issues by natural language or explicit JQL (prefix with JQL:). "
        "Returns issue keys, status, assignee, priority, updated timestamp, and URL."
    )

    MAX_RESULTS = 5
    HTTP_TIMEOUT = 15.0

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

    async def _search(self, query: str) -> tuple[list[dict[str, str]], str | None]:
        if not self._is_configured():
            return [], (
                "Jira search is not configured. "
                "Set JIRA_BASE_URL and auth (JIRA_PAT or JIRA_EMAIL + JIRA_API_TOKEN)."
            )

        jql = self._build_jql(query)
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
        results, error = await self._search(query)
        if error:
            return f"Jira search tool error: {error}"
        return self._format_results(results)
