"""
Reranks retrieved chunks. Identity (passthrough) by default.
Swap to Cohere via config — no code changes needed.
"""
from backend.app.agent.state import RAGAgentState
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.pipeline.rerankers.factory import RerankerFactory


async def reranker_node(state: RAGAgentState) -> dict:
    log = logger.bind(node="reranker")

    strategy = settings.reranker_strategy
    reranker = RerankerFactory.create(strategy)
    chunks = state.get("retrieved_chunks", [])

    reranked = await reranker.rerank(
        query=state["original_query"],
        chunks=chunks,
        top_n=settings.reranker_top_n,
    )

    log.info("reranked", before=len(chunks), after=len(reranked), strategy=strategy)

    return {
        "reranked_chunks": reranked,
        "agent_trace": [
            f"Reranked: {len(chunks)} → {len(reranked)} chunks "
            f"(strategy={strategy})"
        ],
    }
