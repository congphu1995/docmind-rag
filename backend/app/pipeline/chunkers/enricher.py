"""
Prepend situating context to each child chunk before embedding.
Anthropic technique — +15-20% retrieval precision at one-time index cost.
"""
import asyncio

from openai import AsyncOpenAI

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.pipeline.base.chunker import Chunk
from backend.app.pipeline.prompts import ENRICHMENT_PROMPT


class ContextEnricher:
    # TODO(week2): Replace direct AsyncOpenAI with BaseLLMClient injection

    def __init__(self):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = "gpt-4o-mini"

    async def enrich_batch(
        self,
        chunks: list[Chunk],
        doc_metadata: dict,
        concurrency: int = 10,
    ) -> list[Chunk]:
        """
        Enrich all child chunks in a document.
        Skips parents and non-text chunks.
        """
        to_enrich = [c for c in chunks if not c.is_parent and c.type == "text"]
        skip = [c for c in chunks if c.is_parent or c.type != "text"]

        if not to_enrich:
            return chunks

        logger.info(
            "enrichment_start",
            chunks=len(to_enrich),
            doc_id=doc_metadata.get("doc_id"),
        )

        semaphore = asyncio.Semaphore(concurrency)
        tasks = [
            self._enrich_one(chunk, doc_metadata, semaphore) for chunk in to_enrich
        ]
        enriched = await asyncio.gather(*tasks, return_exceptions=True)

        final_enriched = []
        for chunk, result in zip(to_enrich, enriched):
            if isinstance(result, Exception):
                logger.warning(
                    "enrichment_failed", chunk_id=chunk.chunk_id, error=str(result)
                )
                final_enriched.append(chunk)
            else:
                final_enriched.append(result)

        logger.info("enrichment_done", doc_id=doc_metadata.get("doc_id"))
        return skip + final_enriched

    async def _enrich_one(
        self,
        chunk: Chunk,
        doc_metadata: dict,
        semaphore: asyncio.Semaphore,
    ) -> Chunk:
        async with semaphore:
            prompt = ENRICHMENT_PROMPT.format(
                doc_title=doc_metadata.get("title", chunk.doc_name),
                doc_type=doc_metadata.get("doc_type", "document"),
                section=chunk.section or "Unknown",
                chunk_text=chunk.content_raw[:500],
            )

            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=80,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )

            context_sentence = response.choices[0].message.content.strip()
            enriched_content = f"{context_sentence}\n\n{chunk.content_raw}"

            return Chunk(
                **{k: v for k, v in chunk.__dict__.items() if k != "content"},
                content=enriched_content,
            )
