from backend.app.pipeline.base.reranker import BaseReranker


class IdentityReranker(BaseReranker):
    """Passthrough reranker — returns chunks unchanged. Default until Cohere is added."""

    async def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_n: int,
    ) -> list[dict]:
        return chunks[:top_n]
