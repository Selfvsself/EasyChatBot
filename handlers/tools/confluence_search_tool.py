from __future__ import annotations

import json
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, ValidationError

from core.config import settings
from .base_tool import BaseTool


class ConfluenceSearchTool(BaseTool):
    agent_tool_name = "confluence_search"
    agent_tool_description = (
        "Search Confluence pages by natural language or explicit CQL (prefix with CQL:). "
        "The tool can inspect page content and refine search queries automatically."
    )

    MAX_RESULTS = 5
    HTTP_TIMEOUT = 15.0
    MAX_REFINE_ATTEMPTS = 3
    MAX_PAGE_TEXT_CHARS = 12000

    class QueryPlan(BaseModel):
        cql: str
        rationale: str | None = None

    class AnalysisResult(BaseModel):
        answer: str
        is_complete: bool = True
        is_correct: bool = True
        needs_retry: bool = False
        retry_query: str | None = None
        evidence: list[str] = Field(default_factory=list)

    @staticmethod
    def _normalize_site_base(url: str) -> str:
        base = (url or "").strip().rstrip("/")
        if base.endswith("/wiki"):
            return base[:-5]
        return base

    @staticmethod
    def _join_wiki_url(site_base: str, webui: str) -> str:
        if not webui:
            return ""
        if webui.startswith(("http://", "https://")):
            return webui
        if webui.startswith("/wiki/"):
            return f"{site_base}{webui}"
        if webui.startswith("/"):
            return f"{site_base}/wiki{webui}"
        return f"{site_base}/wiki/{webui}"

    def __init__(self, llm_client, message_repo):
        super().__init__(llm_client, message_repo)
        raw_base = settings.CONFLUENCE_BASE_URL or settings.JIRA_BASE_URL or ""
        self.site_base = self._normalize_site_base(raw_base)
        self.wiki_base = f"{self.site_base}/wiki" if self.site_base else ""
        self.api_v1_base = f"{self.wiki_base}/rest/api" if self.wiki_base else ""
        self.email = settings.CONFLUENCE_EMAIL or settings.JIRA_EMAIL or ""
        self.api_token = settings.CONFLUENCE_API_TOKEN or settings.JIRA_API_TOKEN or ""
        self.pat = settings.CONFLUENCE_PAT or settings.JIRA_PAT or ""
        self.space_key = settings.CONFLUENCE_SPACE_KEY or ""

    def _is_configured(self) -> bool:
        has_auth = bool(self.pat) or (bool(self.email) and bool(self.api_token))
        return bool(self.site_base) and has_auth

    def _build_cql(self, query: str) -> str:
        clean_query = (query or "").strip()
        if clean_query.upper().startswith("CQL:"):
            return clean_query[4:].strip()

        escaped = clean_query.replace('"', '\\"')
        space_filter = f'space = "{self.space_key}" AND ' if self.space_key else ""
        return f'{space_filter}type = page AND text ~ "{escaped}" ORDER BY lastmodified DESC'

    def _extract_terms_from_cql(self, cql: str) -> str:
        quoted = re.findall(r'"([^"]+)"', cql or "")
        if quoted:
            return " ".join(part.strip() for part in quoted if part.strip())
        return (cql or "").strip()

    def _sanitize_cql(self, cql: str) -> str:
        text = (cql or "").strip()
        if not text:
            return self._build_cql("")

        # common invalid alias from LLM output
        text = re.sub(r"\bcontent\s*~", "text ~", text, flags=re.IGNORECASE)

        # ensure space key is quoted: space = RE -> space = "RE"
        text = re.sub(
            r'\bspace\s*=\s*([A-Za-z0-9_-]+)\b',
            lambda m: f'space = "{m.group(1)}"',
            text,
            flags=re.IGNORECASE,
        )

        has_type_filter = bool(re.search(r"\btype\s*=", text, flags=re.IGNORECASE))
        if not has_type_filter:
            text = f"type = page AND ({text})"

        has_order = bool(re.search(r"\border\s+by\b", text, flags=re.IGNORECASE))
        if not has_order:
            text = f"{text} ORDER BY lastmodified DESC"

        return text

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

    @staticmethod
    def _clean_html_text(raw_html: str) -> str:
        if not raw_html:
            return ""
        soup = BeautifulSoup(raw_html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", text).strip()

    async def _plan_cql_with_llm(self, user_query: str) -> str:
        plan_system = (
            "You generate CQL for Confluence page search. "
            "Return JSON only with keys: cql, rationale. "
            "Prefer focused CQL aligned with the user intent."
        )
        prompt = self.build_prompt(
            system=plan_system,
            history=[],
            text=(
                f"User query:\n{user_query}\n\n"
                f"Space key: {self.space_key or 'not set'}\n"
                "Return JSON only."
            ),
        )
        raw = await self.llm.chat(prompt)
        try:
            payload = json.loads(self._strip_fences(raw))
            model = self.QueryPlan.model_validate(payload)
            return model.cql.strip()
        except (json.JSONDecodeError, ValidationError, TypeError):
            return self._build_cql(user_query)

    async def _search_by_cql(self, cql: str) -> tuple[list[dict[str, str]], str | None]:
        if not self._is_configured():
            return [], (
                "Confluence search is not configured. "
                "Set CONFLUENCE_BASE_URL and auth (CONFLUENCE_PAT or CONFLUENCE_EMAIL + CONFLUENCE_API_TOKEN)."
            )

        headers, auth = self._auth_headers()
        endpoint = f"{self.api_v1_base}/search"
        primary_cql = self._sanitize_cql(cql)
        fallback_terms = self._extract_terms_from_cql(primary_cql)
        fallback_cql = self._build_cql(fallback_terms)

        cql_candidates = [primary_cql]
        if fallback_cql and fallback_cql not in cql_candidates:
            cql_candidates.append(fallback_cql)

        last_error = "unknown error"
        async with httpx.AsyncClient(timeout=self.HTTP_TIMEOUT) as client:
            for cql_candidate in cql_candidates:
                params = {"cql": cql_candidate, "limit": self.MAX_RESULTS}
                try:
                    response = await client.get(endpoint, params=params, headers=headers, auth=auth)
                    response.raise_for_status()
                    payload = response.json()
                    return self._extract_search_results(payload), None
                except httpx.HTTPStatusError as exc:
                    body = ""
                    try:
                        body = exc.response.text
                    except Exception:
                        body = ""
                    last_error = f"{exc} | body={body[:400]}"
                    continue
                except Exception as exc:
                    last_error = str(exc)
                    continue

        return [], f"Confluence search failed: {last_error}"

    def _extract_search_results(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        results = payload.get("results") or []
        extracted: list[dict[str, str]] = []

        for item in results:
            content = item.get("content") or {}
            page_id = str(content.get("id") or "")
            title = content.get("title") or ""
            excerpt_html = item.get("excerpt") or ""
            excerpt = self._clean_html_text(excerpt_html)
            links = content.get("_links") or {}
            webui = links.get("webui") or ""
            url = self._join_wiki_url(self.site_base, webui)

            if not page_id or not title:
                continue

            extracted.append(
                {
                    "id": page_id,
                    "title": title,
                    "url": url,
                    "excerpt": excerpt,
                }
            )

        return extracted[:self.MAX_RESULTS]

    async def _fetch_page_details(
        self,
        page_id: str,
        fallback_title: str,
        stage_callback=None,
    ) -> dict[str, str] | None:
        if stage_callback:
            await stage_callback(
                stage="confluence_page_opening",
                metadata={"id": page_id, "title": fallback_title, "url": f"{self.wiki_base}/pages/viewpage.action?pageId={page_id}"},
            )

        headers, auth = self._auth_headers()
        endpoint = f"{self.api_v1_base}/content/{page_id}"
        params = {"expand": "body.storage,space,version"}

        try:
            async with httpx.AsyncClient(timeout=self.HTTP_TIMEOUT) as client:
                response = await client.get(endpoint, params=params, headers=headers, auth=auth)
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return None

        title = payload.get("title") or fallback_title
        links = payload.get("_links") or {}
        webui = links.get("webui") or ""
        url = self._join_wiki_url(self.site_base, webui)
        space = ((payload.get("space") or {}).get("key")) or ""
        version = str(((payload.get("version") or {}).get("number")) or "")
        body_html = (((payload.get("body") or {}).get("storage") or {}).get("value")) or ""
        text = self._clean_html_text(body_html)[:self.MAX_PAGE_TEXT_CHARS]

        return {
            "id": page_id,
            "title": title,
            "url": url,
            "space": space,
            "version": version,
            "text": text,
        }

    async def _fetch_many_page_details(self, pages: list[dict[str, str]], stage_callback=None) -> list[dict[str, str]]:
        detailed: list[dict[str, str]] = []
        for page in pages:
            page_id = page.get("id")
            if not page_id:
                continue
            data = await self._fetch_page_details(page_id, page.get("title", ""), stage_callback=stage_callback)
            if data:
                detailed.append(data)
        return detailed

    async def _analyze_with_llm(
        self,
        user_query: str,
        cql: str,
        pages: list[dict[str, str]],
    ) -> AnalysisResult:
        if not pages:
            return self.AnalysisResult(
                answer="No relevant Confluence page details were found.",
                is_complete=False,
                is_correct=True,
                needs_retry=True,
                retry_query=f"{user_query} how-to guide",
                evidence=[],
            )

        blocks = []
        for idx, page in enumerate(pages, start=1):
            blocks.append(
                f"[Page {idx}] {page['title']}\n"
                f"ID: {page['id']}; Space: {page['space']}; Version: {page['version']}\n"
                f"URL: {page['url']}\n"
                f"TEXT:\n{page['text']}"
            )

        system = (
            "You are a Confluence analyst. "
            "Return JSON only with keys: answer, is_complete, is_correct, needs_retry, retry_query, evidence. "
            "evidence must be a list of page URLs or short facts. "
            "Set needs_retry=true only if data is insufficient or contradictory."
        )
        prompt = self.build_prompt(
            system=system,
            history=[],
            text=(
                f"User query:\n{user_query}\n\n"
                f"CQL used:\n{cql}\n\n"
                "Page details:\n\n"
                + "\n\n".join(blocks)
            ),
        )

        raw = await self.llm.chat(prompt)
        try:
            payload = json.loads(self._strip_fences(raw))
            return self.AnalysisResult.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError):
            return self.AnalysisResult(
                answer=raw.strip() or "Could not produce a Confluence analysis answer.",
                is_complete=True,
                is_correct=True,
                needs_retry=False,
                retry_query=None,
                evidence=[],
            )

    async def _best_effort_answer_with_llm(
        self,
        user_query: str,
        cql: str,
        pages: list[dict[str, str]],
    ) -> str:
        if not pages:
            return "No relevant Confluence page details were found."

        blocks = []
        for idx, page in enumerate(pages, start=1):
            blocks.append(
                f"[Page {idx}] {page['title']}\n"
                f"ID: {page['id']}; Space: {page['space']}; Version: {page['version']}\n"
                f"URL: {page['url']}\n"
                f"TEXT:\n{page['text']}"
            )

        prompt = self.build_prompt(
            system=(
                "You are a Confluence analyst. Provide the best possible answer from available page data. "
                "If data is partial, explicitly mention uncertainty. Return plain text only."
            ),
            history=[],
            text=(
                f"User query:\n{user_query}\n\n"
                f"CQL used:\n{cql}\n\n"
                "Page details:\n\n"
                + "\n\n".join(blocks)
            ),
        )
        answer = await self.llm.chat(prompt)
        return (answer or "").strip() or "Could not produce a Confluence answer."

    async def processing(self, chat, app, text):
        pages, error = await self._search_by_cql(self._build_cql(text))
        if error:
            return f"Confluence search tool error: {error}"

        if not pages:
            return "No Confluence pages found."

        lines = []
        for idx, page in enumerate(pages, start=1):
            lines.append(
                f"{idx}. {page['title']}\n"
                f"URL: {page['url']}\n"
                f"Excerpt: {page['excerpt'] or 'n/a'}"
            )
        return (
            "Confluence page search results are provided below. "
            "Use this as factual context and cite page URLs when relevant.\n\n"
            + "\n\n".join(lines)
        )

    async def run_for_agent(self, query: str, chat=None, app=None, stage_callback=None) -> str:
        working_query = query
        last_analysis: ConfluenceSearchTool.AnalysisResult | None = None
        last_cql = self._build_cql(query)
        last_pages: list[dict[str, str]] = []

        for attempt in range(1, self.MAX_REFINE_ATTEMPTS + 1):
            is_last_attempt = attempt == self.MAX_REFINE_ATTEMPTS

            if stage_callback:
                await stage_callback(
                    stage="confluence_query_planning",
                    metadata={"attempt": attempt, "query": working_query},
                )

            cql = await self._plan_cql_with_llm(working_query)
            last_cql = cql or self._build_cql(working_query)

            if stage_callback:
                await stage_callback(
                    stage="confluence_query_running",
                    metadata={"attempt": attempt, "cql": last_cql},
                )

            pages, error = await self._search_by_cql(last_cql)
            if error:
                return f"Confluence search tool error: {error}"

            detailed = await self._fetch_many_page_details(pages, stage_callback=stage_callback)
            last_pages = detailed

            if stage_callback and detailed:
                await stage_callback(
                    stage="sources_used",
                    metadata={
                        "sources": [{"title": p["title"], "url": p["url"]} for p in detailed if p.get("url")]
                    },
                )

            if is_last_attempt:
                best_effort = await self._best_effort_answer_with_llm(query, last_cql, detailed)
                pages_line = ", ".join(p["id"] for p in detailed) if detailed else "none"
                return (
                    f"CQL used: {last_cql}\n"
                    f"Pages inspected: {pages_line}\n"
                    f"Evidence: n/a\n\n"
                    f"Answer:\n{best_effort}"
                )

            if stage_callback:
                await stage_callback(
                    stage="confluence_validating",
                    metadata={"attempt": attempt, "pages": [p["id"] for p in detailed]},
                )

            analysis = await self._analyze_with_llm(query, last_cql, detailed)
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
            pages_line = ", ".join(p["id"] for p in detailed) if detailed else "none"
            return (
                f"CQL used: {last_cql}\n"
                f"Pages inspected: {pages_line}\n"
                f"Evidence: {evidence_line}\n\n"
                f"Answer:\n{analysis.answer}"
            )

        if last_analysis:
            evidence_line = ", ".join(last_analysis.evidence) if last_analysis.evidence else "n/a"
            pages_line = ", ".join(p["id"] for p in last_pages) if last_pages else "none"
            return (
                f"CQL used: {last_cql}\n"
                f"Pages inspected: {pages_line}\n"
                f"Evidence: {evidence_line}\n\n"
                f"Answer:\n{last_analysis.answer}"
            )

        return "Confluence search completed but no answer could be produced."
