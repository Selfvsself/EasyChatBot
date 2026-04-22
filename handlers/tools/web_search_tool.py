from __future__ import annotations

import asyncio
import json
import re
import random
from typing import Any

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from pydantic import BaseModel, Field, ValidationError

from .base_tool import BaseTool


class WebSearchTool(BaseTool):
    agent_tool_name = "web_search"
    agent_tool_description = (
        "Search the web for current/public information and return grounded snippets with URLs. "
        "Use for facts, external references, and recent context."
    )

    MAX_RESULTS = 5
    MAX_PAGES = 3
    MAX_PAGE_CHARS = 6000
    HTTP_TIMEOUT = 10.0
    MAX_REFINE_ATTEMPTS = 3

    class QueryPlan(BaseModel):
        reasoning: str
        search_queries: list[str] | None = None

    class AnalysisResult(BaseModel):
        answer: str
        is_complete: bool = True
        is_correct: bool = True
        needs_retry: bool = False
        retry_query: str | None = None
        evidence: list[str] = Field(default_factory=list)

    async def search_web(self, query: str) -> list[dict[str, str]]:
        try:
            raw_results = await asyncio.to_thread(self._search_sync, query)
            return self._extract_results(raw_results)
        except Exception:
            return []

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

    def _search_sync(self, query: str) -> list[dict[str, Any]]:
        with DDGS(timeout=10) as ddgs:
            results = ddgs.text(
                query=query,
                max_results=self.MAX_RESULTS,
                region="ru-ru",
                safesearch="on",
                backend="api"
            )
        return list(results or [])

    def _extract_results(self, payload: list[dict[str, Any]]) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        for item in payload:
            title = item.get("title")
            url = item.get("href") or item.get("url")
            snippet = item.get("body") or item.get("snippet")
            if not title or not url:
                continue

            results.append({"title": title, "url": url, "snippet": snippet or ""})

        return results[:self.MAX_RESULTS]

    async def _fetch_page_text(self, client: httpx.AsyncClient, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            return ""

        try:
            response = await client.get(
                url,
                follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    )
                }
            )
            response.raise_for_status()
        except Exception:
            return ""

        content_type = (response.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:self.MAX_PAGE_CHARS]

    async def _load_sources(self, results: list[dict[str, str]]) -> list[dict[str, str]]:
        if not results:
            return []

        sources = results[:self.MAX_PAGES]
        async with httpx.AsyncClient(timeout=self.HTTP_TIMEOUT) as client:
            tasks = [self._fetch_page_text(client, item["url"]) for item in sources]
            page_texts = await asyncio.gather(*tasks, return_exceptions=False)

        loaded: list[dict[str, str]] = []
        for item, page_text in zip(sources, page_texts):
            if not page_text:
                continue
            loaded.append(
                {
                    "title": item["title"],
                    "url": item["url"],
                    "text": page_text
                }
            )
        return loaded

    async def _load_sources_for_agent(self, results: list[dict[str, str]], stage_callback=None) -> list[dict[str, str]]:
        if not results:
            return []

        sources = results[:self.MAX_PAGES]
        loaded: list[dict[str, str]] = []

        async with httpx.AsyncClient(timeout=self.HTTP_TIMEOUT) as client:
            for item in sources:
                if stage_callback:
                    await stage_callback(
                        stage="site_reading",
                        metadata={
                            "url": item["url"],
                            "title": item["title"],
                        },
                    )

                page_text = await self._fetch_page_text(client, item["url"])
                if not page_text:
                    continue

                loaded.append(
                    {
                        "title": item["title"],
                        "url": item["url"],
                        "text": page_text
                    }
                )

        if stage_callback and loaded:
            await stage_callback(
                stage="sources_used",
                metadata={
                    "sources": [{"title": src["title"], "url": src["url"]} for src in loaded]
                },
            )

        return loaded

    async def _summarize_sources(self, user_query: str, sources: list[dict[str, str]]) -> str:
        if not sources:
            return "No source text could be extracted from search results."

        blocks = []
        for idx, src in enumerate(sources, start=1):
            blocks.append(
                f"[Source {idx}] {src['title']}\n"
                f"URL: {src['url']}\n"
                f"TEXT:\n{src['text']}"
            )

        summarizer_system = (
            "Summarize web sources for grounding an assistant answer. "
            "Return concise bullet points with factual claims only. "
            "Include source labels like [Source 1] per bullet. "
            "If sources conflict, call out the conflict."
        )
        summarizer_text = (
            f"User query: {user_query}\n\n"
            "Sources:\n\n"
            + "\n\n".join(blocks)
        )
        summarizer_prompt = self.build_prompt(
            system=summarizer_system,
            history=[],
            text=summarizer_text,
        )
        return await self.llm.chat(summarizer_prompt)

    @staticmethod
    def _strip_fences(text: str) -> str:
        return re.sub(r"```(?:json)?\n?|```", "", text or "").strip()

    async def _plan_search_query_with_llm(self, user_query: str, history: list = None, context: str = None) -> list[str]:
        plan_system = (
            "Role: "
            "You are an expert Search Query Engineer. "
            "Your task is to analyze the user's current request and, using the provided "
            "context and history, generate a set of precise search queries to find the most relevant information. "
            "Input Data: "
            "1. User Request: The current prompt from the user. "
            "2. User Context (Facts): Key details, entities, and long-term facts about the user's projects or preferences. "
            "3. Message History: Recent messages to track the specific topic evolution. "
            "Operational Rules: "
            "- De-contextualization: Convert ambiguous pronouns (it, they, that project, his company) into specific names or terms found in the Context or History. "
            "- Multi-perspective Search: If the request is complex, break it down into 2-3 different queries (e.g., one for technical specs, one for market trends). "
            "- Efficiency: Do not search for things already clearly defined in the \"User Context\" or \"History\". "
            "- No Explanations: Your output must be only the list of queries. "
            "Output Format: "
            "Return a valid JSON object. "
            "JSON Schema: "
            "{ "
            "\"reasoning\": \"Briefly explain why these specific terms were chosen based on the history/context\", "
            "\"search_queries\": [\"query 1\", \"query 2\", \"query 3\"] "
            "}"
        )
        formatted_history = self._build_messages_block(history)
        plan_prompt = self.build_prompt(
            system=plan_system,
            history=[],
            text=(f"User query:\n{user_query}\n\n"
                  f"User Context (Facts):\n{context}\n\n"
                  f"Message History:\n{formatted_history}\n\n"
                  "Return JSON only."),
        )
        raw = await self.llm.chat(plan_prompt, response_format="json")
        try:
            payload = json.loads(self._strip_fences(raw))
            plan = self.QueryPlan.model_validate(payload)
            return plan.search_queries or [user_query]
        except (json.JSONDecodeError, ValidationError, TypeError):
            return [user_query]

    async def _analyze_sources_with_llm(
        self,
        user_query: str,
        search_query: str,
        web_results: list[dict[str, str]],
        sources: list[dict[str, str]],
    ) -> AnalysisResult:
        snippet_block = []
        for idx, item in enumerate(web_results, start=1):
            snippet_block.append(
                f"[Result {idx}] {item['title']}\nURL: {item['url']}\nSnippet: {item['snippet']}"
            )

        source_block = []
        for idx, item in enumerate(sources, start=1):
            source_block.append(
                f"[Source {idx}] {item['title']}\nURL: {item['url']}\nTEXT:\n{item['text']}"
            )

        analysis_system = (
            "You are a source-grounded analyst. "
            "Return JSON only with keys: answer, is_complete, is_correct, needs_retry, retry_query, evidence. "
            "evidence is a list of URLs or short facts. "
            "Set needs_retry=true only if sources are insufficient or contradictory."
        )
        analysis_prompt = self.build_prompt(
            system=analysis_system,
            history=[],
            text=(
                f"User query:\n{user_query}\n\n"
                f"Search query used:\n{search_query}\n\n"
                "Search results:\n"
                + ("\n\n".join(snippet_block) if snippet_block else "none")
                + "\n\nExtracted source texts:\n"
                + ("\n\n".join(source_block) if source_block else "none")
            ),
        )

        raw = await self.llm.chat(analysis_prompt)
        try:
            payload = json.loads(self._strip_fences(raw))
            return self.AnalysisResult.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError):
            return self.AnalysisResult(
                answer=raw.strip() or "Could not produce a grounded answer.",
                is_complete=True,
                is_correct=True,
                needs_retry=False,
                retry_query=None,
                evidence=[],
            )

    async def _best_effort_answer_with_llm(
        self,
        user_query: str,
        search_query: str,
        web_results: list[dict[str, str]],
        sources: list[dict[str, str]],
    ) -> str:
        snippet_block = []
        for idx, item in enumerate(web_results, start=1):
            snippet_block.append(
                f"[Result {idx}] {item['title']}\nURL: {item['url']}\nSnippet: {item['snippet']}"
            )

        source_block = []
        for idx, item in enumerate(sources, start=1):
            source_block.append(
                f"[Source {idx}] {item['title']}\nURL: {item['url']}\nTEXT:\n{item['text']}"
            )

        prompt = self.build_prompt(
            system=(
                "Provide the best possible factual answer from available web data. "
                "If sources are incomplete, clearly mention uncertainty. Return plain text only."
            ),
            history=[],
            text=(
                f"User query:\n{user_query}\n\n"
                f"Search query used:\n{search_query}\n\n"
                "Search results:\n"
                + ("\n\n".join(snippet_block) if snippet_block else "none")
                + "\n\nExtracted source texts:\n"
                + ("\n\n".join(source_block) if source_block else "none")
            ),
        )
        answer = await self.llm.chat(prompt)
        return (answer or "").strip() or "Could not produce a web answer."

    def format_results(self, results: list[dict[str, str]]) -> str:
        if not results:
            return "No web search results."

        lines = []
        for idx, result in enumerate(results, start=1):
            lines.append(
                f"{idx}. {result['title']}\n"
                f"URL: {result['url']}\n"
                f"Fact: {result['snippet']}"
            )
        return "\n\n".join(lines)

    async def processing(self, chat, app, text):
        web_results = await self.search_web(text)
        web_context = self.format_results(web_results)
        loaded_sources = await self._load_sources(web_results)
        summarized_context = await self._summarize_sources(text, loaded_sources)

        return (
            "Web search results are provided below. "
            "The source text was extracted and summarized. "
            "Use this context for factual grounding and cite URLs when relevant. "
            "If data is insufficient, say so clearly.\n\n"
            f"Summarized source context:\n{summarized_context}\n\n"
            "Search results list:\n"
            f"{web_context}"
        )

    async def run_for_agent(self, query: str, history=None, context=None, stage_callback=None) -> str:
        if stage_callback:
            await stage_callback(
                stage="web_query_planning",
                metadata={"attempt": 1, "query": query},
            )

        search_queries = await self._plan_search_query_with_llm(query, history, context)
        search_query = random.choice(search_queries)

        if stage_callback:
            await stage_callback(
                stage="web_search_running",
                metadata={"attempt": 1, "query": search_query},
            )

        web_results = await self.search_web(search_query)
        if not web_results:
            return "No web search results."

        loaded_sources = await self._load_sources_for_agent(web_results, stage_callback=stage_callback)
        compact_sources = []
        for idx, src in enumerate(loaded_sources, start=1):
            compact_sources.append(
                f"[Source {idx}] {src['title']}\nURL: {src['url']}\nTEXT: {src['text']}"
            )

        return (
            f"Search query used: {search_query}\n\n"
            f"Search results:\n{self.format_results(web_results)}\n\n"
            + ("Extracted source text:\n" + "\n\n".join(compact_sources) if compact_sources else "No source text extracted.")
        )
