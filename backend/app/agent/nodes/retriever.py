"""
Adaptive retriever with retry loop.
1. Dense search in Qdrant (child chunks)
2. Assess quality score
3. Retry with expanded query if quality < threshold
4. Fetch parent chunks from PostgreSQL for richer LLM context
"""
from sqlalchemy import select

from backend.app.agent.state import RAGAgentState
from backend.app.core.config import settings
from backend.app.core.database import AsyncSessionLocal
from backend.app.core.logging import logger
from backend.app.models.document import ParentChunk
from backend.app.pipeline.embedders.openai_embedder import OpenAIEmbedder
from backend.app.pipeline.llm.factory import LLMFactory
from backend.app.vectorstore.qdrant_client import QdrantWrapper


MAX_ATTEMPTS = 3


def _assess_quality(results: list) -> float:
    """Average score of top-5 results. 0.0 if empty."""
    if not results:
        return 0.0
    scores = [r.score for r in results[:5]]
    return sum(scores) / len(scores)


async def _fetch_parents(child_results: list) -> list[dict]:
    """
    Look up parent chunks from PostgreSQL.
    For each child, return the parent's full content for richer LLM context.
    Atomic chunks (no parent_id) pass through directly.
    """
    parent_ids = {
        r.payload.get("parent_id")
        for r in child_results
        if r.payload.get("parent_id")
    }

    parents = {}
    if parent_ids:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(ParentChunk).where(ParentChunk.chunk_id.in_(parent_ids))
            )
            parents = {p.chunk_id: p for p in result.scalars().all()}

    chunks = []
    seen_parents: set[str] = set()

    for r in child_results:
        parent_id = r.payload.get("parent_id")

        if parent_id and parent_id in parents and parent_id not in seen_parents:
            parent = parents[parent_id]
            chunks.append({
                "content": parent.content_raw,
                "content_markdown": parent.content_markdown,
                "doc_id": parent.doc_id,
                "doc_name": r.payload.get("doc_name", ""),
                "page": parent.page,
                "section": parent.section,
                "type": parent.type,
                "score": r.score,
                "chunk_id": parent.chunk_id,
            })
            seen_parents.add(parent_id)

        elif not parent_id:
            # Atomic chunk (table, figure) — no parent, use child directly
            chunks.append({
                "content": r.payload.get("content_raw", ""),
                "content_markdown": r.payload.get("content_markdown"),
                "doc_id": r.payload.get("doc_id", ""),
                "doc_name": r.payload.get("doc_name", ""),
                "page": r.payload.get("page", 0),
                "section": r.payload.get("section", ""),
                "type": r.payload.get("type", "text"),
                "score": r.score,
                "chunk_id": r.payload.get("chunk_id", ""),
            })

    return chunks


async def retriever_node(state: RAGAgentState) -> dict:
    log = logger.bind(node="retriever")

    embedder = OpenAIEmbedder()
    qdrant = QdrantWrapper()

    # Use HyDE query if available, otherwise rewritten query
    query_text = state.get("hyde_query") or state["rewritten_query"] or state["original_query"]

    # Build Qdrant filters
    filters = {}
    if state.get("doc_ids"):
        filters["doc_ids"] = state["doc_ids"]
    if state.get("detected_language") and state["detected_language"] != "en":
        filters["language"] = state["detected_language"]

    quality = 0.0
    results = []
    attempt = 0

    for attempt in range(MAX_ATTEMPTS):
        vector = await embedder.embed_single(query_text)
        results = await qdrant.search(
            vector=vector,
            top_k=settings.retrieval_top_k,
            filters=filters if filters else None,
        )

        quality = _assess_quality(results)
        log.info(
            "retrieval_attempt",
            attempt=attempt + 1,
            results=len(results),
            quality=round(quality, 3),
        )

        if quality >= settings.retrieval_quality_threshold or attempt == MAX_ATTEMPTS - 1:
            break

        # Retry: expand query
        if attempt == 0:
            llm = LLMFactory.create_mini()
            query_text = await llm.complete(
                messages=[{
                    "role": "user",
                    "content": f"Rephrase this for better document search. "
                               f"Add synonyms and related terms. "
                               f"Output ONLY the query:\n\n{state['original_query']}",
                }],
                max_tokens=100,
                temperature=0.3,
            )
            query_text = query_text.strip()
        elif attempt == 1 and state.get("sub_questions"):
            # Attempt 3: try first sub-question
            query_text = state["sub_questions"][0]

    # Fetch parent chunks from PostgreSQL
    chunks = await _fetch_parents(results)

    log.info(
        "retrieval_done",
        attempts=attempt + 1,
        quality=round(quality, 3),
        chunks=len(chunks),
    )

    return {
        "retrieved_chunks": chunks,
        "retrieval_attempts": attempt + 1,
        "retrieval_quality": quality,
        "agent_trace": [
            f"Retrieved {len(chunks)} chunks in {attempt + 1} attempt(s) "
            f"(quality={quality:.2f})"
        ],
    }
