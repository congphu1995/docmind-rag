"""
Rewrites query for optimal retrieval.
Conditional HyDE: only for analytical and multi_hop queries.
"""

import time

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from backend.app.agent.llm import get_mini_model
from backend.app.agent.prompts import HYDE_PROMPT, QUERY_REWRITE_PROMPT
from backend.app.agent.state import RAGAgentState
from backend.app.core.config import settings
from backend.app.core.logging import logger


def _should_use_hyde(query: str, query_type: str) -> bool:
    if query_type in ("greeting", "general"):
        return False
    if query_type in ("multi_hop", "analytical"):
        return True
    return False


async def query_rewriter(state: RAGAgentState, config: RunnableConfig = None) -> dict:
    query = state["original_query"]
    query_type = state["query_type"]
    log = logger.bind(node="query_rewriter")

    llm = get_mini_model().bind(temperature=0, max_tokens=150)

    rewrite_result = await llm.ainvoke(
        [
            HumanMessage(content=QUERY_REWRITE_PROMPT.format(query=query, context="")),
        ],
        config=config,
    )
    rewritten = rewrite_result.content.strip()

    hyde_query = ""
    hyde_used = False
    trace_parts = [f"Rewritten: '{rewritten}'"]

    if _should_use_hyde(query, query_type):
        start = time.perf_counter()
        hyde_llm = get_mini_model().bind(
            temperature=0.3, max_tokens=settings.hyde_max_tokens
        )
        hyde_result = await hyde_llm.ainvoke(
            [HumanMessage(content=HYDE_PROMPT.format(query=rewritten))],
            config=config,
        )
        hyde_query = hyde_result.content.strip()
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
