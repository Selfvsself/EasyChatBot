from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS


class WebSearchTool:

    def __init__(self, max_results=10, max_page_chars=6000):
        self.MAX_RESULTS = max_results
        self.MAX_PAGE_CHARS = max_page_chars
        self.HTTP_TIMEOUT = 10.0

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
                backend="yandex, duckduckgo, brave, yahoo"
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

    async def _load_page_source(self, web_result: dict[str, str]) -> dict[str, str]:
        if not web_result:
            return {}

        loaded = {}

        async with httpx.AsyncClient(timeout=self.HTTP_TIMEOUT) as client:
            page_text = await self._fetch_page_text(client, web_result["url"])
            if page_text:
                loaded: dict[str, str] = {
                    "title": web_result["title"],
                    "url": web_result["url"],
                    "text": page_text
                }
        return loaded

    async def search(self, query: str) -> list[dict[str, str]]:
        web_results = await self.search_web(query)
        if not web_results:
            return []
        return web_results

    async def get_page_source(self, web_result: dict[str, str]):
        content = await self._load_page_source(web_result)
        return content
