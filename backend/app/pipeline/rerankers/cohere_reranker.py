import cohere

from backend.app.core.logging import logger
from backend.app.pipeline.base.reranker import BaseReranker


class CohereReranker(BaseReranker):
    """Cross-encoder reranker using Cohere Rerank v3."""

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v3.0",
    ):
        self._client = cohere.ClientV2(api_key=api_key)
        self._model = model

    async def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_n: int,
    ) -> list[dict]:
        if not chunks:
            return []

        documents = [c.get("content", "") for c in chunks]

        try:
            response = self._client.rerank(
                model=self._model,
                query=query,
                documents=documents,
                top_n=top_n,
            )
        except Exception as e:
            logger.warning("cohere_rerank_failed", error=str(e))
            return chunks[:top_n]

        reranked = []
        for result in response.results:
            chunk = {**chunks[result.index]}
            chunk["score"] = result.relevance_score
            reranked.append(chunk)

        return reranked
