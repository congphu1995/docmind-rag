"""
Rewrites query for optimal retrieval.
Conditional HyDE: only for analytical and multi_hop queries.
Decision based on query_analyzer LLM classification.
"""
import time

from backend.app.agent.prompts import HYDE_PROMPT, QUERY_REWRITE_PROMPT
from backend.app.agent.state import RAGAgentState
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.pipeline.llm.factory import LLMFactory


def _should_use_hyde(query: str, query_type: str) -> bool:
    """
    Decide whether HyDE is beneficial for this query.
    Relies on query_analyzer LLM classification — no manual regex rules.
    """
    if query_type in ("greeting", "general"):
        return False
    if query_type in ("multi_hop", "analytical"):
        return True
    return False


async def query_rewriter(state: RAGAgentState) -> dict:
    query = state["original_query"]
    query_type = state["query_type"]
    log = logger.bind(node="query_rewriter")

    llm = LLMFactory.create_mini()

    # Rewrite: expand abbreviations, resolve references
    rewritten = await llm.complete(
        messages=[
            {
                "role": "user",
                "content": QUERY_REWRITE_PROMPT.format(
                    query=query,
                    context="",  # TODO: pass conversation history
                ),
            }
        ],
        max_tokens=150,
        temperature=0,
    )
    rewritten = rewritten.strip()

    # Conditional HyDE
    hyde_query = ""
    hyde_used = False
    trace_parts = [f"Rewritten: '{rewritten}'"]

    if _should_use_hyde(query, query_type):
        start = time.perf_counter()
        hyde_query = await llm.complete(
            messages=[
                {"role": "user", "content": HYDE_PROMPT.format(query=rewritten)}
            ],
            max_tokens=settings.hyde_max_tokens,
            temperature=0.3,
        )
        hyde_query = hyde_query.strip()
        hyde_used = True
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        trace_parts.append(
            f"HyDE: used — {query_type} query, "
            f"generated {len(hyde_query.split())} words in {elapsed_ms}ms"
        )
        log.info("hyde_generated", query_type=query_type, elapsed_ms=elapsed_ms)
    else:
        reason = _hyde_skip_reason(query, query_type)
        trace_parts.append(f"HyDE: skipped — {reason}")

    return {
        "rewritten_query": rewritten,
        "hyde_query": hyde_query,
        "hyde_used": hyde_used,
        "agent_trace": ["; ".join(trace_parts)],
    }


def _hyde_skip_reason(query: str, query_type: str) -> str:
    if query_type in ("greeting", "general"):
        return f"{query_type} query"
    return f"{query_type} query — not analytical or multi_hop"
