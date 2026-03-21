"""
Reranks retrieved chunks. Identity (passthrough) by default.
Swap to Cohere via config — no code changes needed.
"""

import time

from backend.app.agent.state import RAGAgentState
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.core.metrics import RERANKER_DURATION
from backend.app.pipeline.rerankers.factory import RerankerFactory


async def reranker_node(state: RAGAgentState) -> dict:
    log = logger.bind(node="reranker")

    strategy = settings.reranker_strategy
    reranker = RerankerFactory.create(strategy)
    chunks = state.get("retrieved_chunks", [])

    start = time.perf_counter()
    reranked = await reranker.rerank(
        query=state["original_query"],
        chunks=chunks,
        top_n=settings.reranker_top_n,
    )
    RERANKER_DURATION.labels(strategy=strategy).observe(time.perf_counter() - start)

    log.info("reranked", before=len(chunks), after=len(reranked), strategy=strategy)

    return {
        "reranked_chunks": reranked,
        "agent_trace": [
            f"Reranked: {len(chunks)} → {len(reranked)} chunks (strategy={strategy})"
        ],
    }
