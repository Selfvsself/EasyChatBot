import asyncio
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS

from .base_tool import BaseTool


class WebSearchTool(BaseTool):
    MAX_RESULTS = 5
    MAX_PAGES = 3
    MAX_PAGE_CHARS = 6000
    HTTP_TIMEOUT = 10.0

    async def search_web(self, query: str) -> list[dict[str, str]]:
        try:
            raw_results = await asyncio.to_thread(self._search_sync, query)
            return self._extract_results(raw_results)
        except Exception:
            return []

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
