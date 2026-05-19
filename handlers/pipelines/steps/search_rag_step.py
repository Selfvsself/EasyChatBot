import logging
import math

import httpx

from core.config import settings
from handlers.pipelines.steps.base_step import BaseStep
from handlers.pipelines.steps.step_context import SearchResult, StepContext
from handlers.pipelines.steps.step_result import StepResult


class SearchRagStep(BaseStep):
    def __init__(self, llm_client, top_chunks: int=1, chunk_size: int=3000, chunk_overlap: int=500):
        super().__init__(llm_client)
        self.top_chunks = top_chunks
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def execute(self, context: StepContext, stage_callback=None) -> StepResult:
        if not context:
            raise ValueError("context is missing or empty")

        search_results = self.get_search_results(context)
        if not search_results:
            return StepResult(context=context, stop=False, success=False)

        filtered_results = await self._maybe_apply_rag(context, search_results, self.top_chunks, self.chunk_size, self.chunk_overlap)
        result_ctx = StepContext.from_context(context)
        result_ctx.search_results = filtered_results
        return StepResult(context=result_ctx, stop=False, success=True)

    def stage(self) -> str:
        return "thinking"

    async def _maybe_apply_rag(self,
                               context: StepContext,
                               search_results: list[SearchResult],
                               top_chunks: int,
                               chunk_size: int,
                               chunk_overlap: int) -> list[SearchResult]:
        try:
            user_query = self.get_next_step_query(context)
            query_embedding = await self._embed_texts([user_query])
            if not query_embedding:
                return search_results

            query_vector = query_embedding[0]
            compressed_results: list[SearchResult] = []
            for result in search_results:
                selected_text = await self._select_best_chunks_for_result(query_vector, result, top_chunks, chunk_size, chunk_overlap)
                compressed_results.append(SearchResult(result.title, result.url, selected_text))

            return compressed_results
        except Exception as e:
            logging.warning(f"RAG prefilter failed, fallback to full text: {e}")
            return search_results

    async def _select_best_chunks_for_result(self,
                                             query_vector: list[float],
                                             result: SearchResult,
                                             top_chunks: int,
                                             chunk_size: int,
                                             chunk_overlap: int) -> str:
        chunks = self._split_text(result.text or "", chunk_size, chunk_overlap)
        if not chunks:
            return result.text or ""

        if len(chunks) <= top_chunks:
            return "\n\n".join(chunks)

        embeddings = await self._embed_texts(chunks)
        if len(embeddings) != len(chunks):
            return result.text or ""

        scored = []
        for idx, vector in enumerate(embeddings):
            similarity = self._cosine_similarity(query_vector, vector)
            scored.append((similarity, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_n = max(1, top_chunks)
        selected_indices = [idx for _, idx in scored[:top_n]]
        selected_indices.sort()
        selected_chunks = [chunks[idx] for idx in selected_indices]
        return "\n\n".join(selected_chunks)

    @staticmethod
    def _split_text(text: str, size: int, chunk_overlap: int) -> list[str]:
        chunk_size = max(200, size)
        overlap = max(0, min(chunk_overlap, chunk_size - 1))
        step = max(1, chunk_size - overlap)

        normalized_text = text.strip()
        if not normalized_text:
            return []

        if len(normalized_text) <= chunk_size:
            return [normalized_text]

        chunks: list[str] = []
        for start in range(0, len(normalized_text), step):
            chunk = normalized_text[start:start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            if start + chunk_size >= len(normalized_text):
                break
        return chunks

    @staticmethod
    async def _embed_texts(texts: list[str]) -> list[list[float]]:
        payload = {
            "model": settings.RAG_EMBEDDING_MODEL,
            "input": texts,
        }
        timeout = httpx.Timeout(30.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(settings.RAG_EMBEDDING_URL, json=payload)
            response.raise_for_status()
            data = response.json()

        embeddings = data.get("embeddings")
        if embeddings and isinstance(embeddings, list):
            return embeddings

        single_embedding = data.get("embedding")
        if single_embedding and isinstance(single_embedding, list):
            return [single_embedding]

        return []

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return -1.0

        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for ai, bi in zip(a, b):
            dot += ai * bi
            norm_a += ai * ai
            norm_b += bi * bi

        if norm_a <= 0.0 or norm_b <= 0.0:
            return -1.0

        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
