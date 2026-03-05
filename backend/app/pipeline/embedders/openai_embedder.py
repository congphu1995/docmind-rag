import asyncio

from openai import AsyncOpenAI

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.pipeline.base.embedder import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):

    def __init__(self):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model
        self._dimensions = settings.embedding_dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        batch_size = 100
        all_vectors = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = await self._embed_batch_with_retry(batch)
            all_vectors.extend(vectors)

        return all_vectors

    async def embed_single(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        return vectors[0]

    async def _embed_batch_with_retry(
        self,
        texts: list[str],
        max_retries: int = 3,
    ) -> list[list[float]]:
        for attempt in range(max_retries):
            try:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2**attempt
                logger.warning(
                    "embedding_retry", attempt=attempt, wait=wait, error=str(e)
                )
                await asyncio.sleep(wait)
