"""
Adaptive retriever with retry loop.
1. Hybrid search in vectorstore (child chunks)
2. Assess quality score
3. Retry with expanded query if quality < threshold
4. Fetch parent chunks for richer LLM context
"""

import time

from backend.app.agent.state import RAGAgentState
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.core.metrics import RETRIEVAL_DURATION
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from backend.app.agent.llm import get_mini_model
from backend.app.pipeline.embedders.openai_embedder import OpenAIEmbedder
from backend.app.vectorstore.factory import VectorStoreFactory


MAX_ATTEMPTS = 3


def _assess_quality(results: list) -> float:
    """Average score of top-5 results. 0.0 if empty."""
    if not results:
        return 0.0
    scores = [r["score"] for r in results[:5]]
    return sum(scores) / len(scores)


async def _fetch_parents(child_results: list, vectorstore) -> list[dict]:
    """
    Look up parent chunks from vectorstore.
    For each child, return the parent's full content for richer LLM context.
    Atomic chunks (no parent_id) pass through directly.
    """
    parent_ids = {
        r["parent_id"] for r in child_results if r.get("parent_id")
    }

    parents = {}
    if parent_ids:
        parent_list = await vectorstore.fetch_parents(list(parent_ids))
        parents = {p["chunk_id"]: p for p in parent_list}

    chunks = []
    seen_parents: set[str] = set()

    for r in child_results:
        parent_id = r.get("parent_id")

        if parent_id and parent_id in parents and parent_id not in seen_parents:
            parent = parents[parent_id]
            chunks.append(
                {
                    "content": parent.get("content_raw", ""),
                    "content_markdown": parent.get("content_markdown"),
                    "doc_id": parent.get("doc_id", ""),
                    "doc_name": r.get("doc_name", ""),
                    "page": parent.get("page", 0),
                    "section": parent.get("section", ""),
                    "type": parent.get("type", "text"),
                    "score": r["score"],
                    "chunk_id": parent["chunk_id"],
                }
            )
            seen_parents.add(parent_id)

        elif not parent_id:
            chunks.append(
                {
                    "content": r.get("content_raw", ""),
                    "content_markdown": r.get("content_markdown"),
                    "doc_id": r.get("doc_id", ""),
                    "doc_name": r.get("doc_name", ""),
                    "page": r.get("page", 0),
                    "section": r.get("section", ""),
                    "type": r.get("type", "text"),
                    "score": r["score"],
                    "chunk_id": r.get("chunk_id", ""),
                }
            )

    return chunks


async def retriever_node(state: RAGAgentState, config: RunnableConfig = None) -> dict:
    log = logger.bind(node="retriever")

    embedder = OpenAIEmbedder()
    vectorstore = VectorStoreFactory.create()

    # Use HyDE query if available, otherwise rewritten query
    query_text = (
        state.get("hyde_query") or state["rewritten_query"] or state["original_query"]
    )

    # Build filters
    filters = {}
    if state.get("doc_ids"):
        filters["doc_ids"] = state["doc_ids"]
    if state.get("detected_language") and state["detected_language"] != "en":
        filters["language"] = state["detected_language"]

    quality = 0.0
    results = []
    attempt = 0

    search_start = time.perf_counter()

    for attempt in range(MAX_ATTEMPTS):
        vector = await embedder.embed_single(query_text)
        results = await vectorstore.search(
            query_vector=vector,
            query_text=query_text,
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

        if (
            quality >= settings.retrieval_quality_threshold
            or attempt == MAX_ATTEMPTS - 1
        ):
            break

        # Retry: expand query
        if attempt == 0:
            llm = get_mini_model().bind(temperature=0.3, max_tokens=100)
            result = await llm.ainvoke(
                [
                    HumanMessage(
                        content=f"Rephrase this for better document search. "
                        f"Add synonyms and related terms. "
                        f"Output ONLY the query:\n\n{state['original_query']}"
                    )
                ],
                config=config,
            )
            query_text = result.content.strip()
        elif attempt == 1 and state.get("sub_questions"):
            # Attempt 3: try first sub-question
            query_text = state["sub_questions"][0]

    # Fetch parent chunks from vectorstore
    chunks = await _fetch_parents(results, vectorstore)

    RETRIEVAL_DURATION.observe(time.perf_counter() - search_start)

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
