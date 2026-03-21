"""
Final generation node. Builds prompt with enriched context, generates answer
with citations, and provides direct_response/direct_llm for non-retrieval paths.
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from backend.app.agent.llm import get_chat_model
from backend.app.agent.prompts import (
    DIRECT_LLM_SYSTEM,
    GENERATION_PROMPT,
    GENERATION_SYSTEM,
    GREETING_RESPONSE,
)
from backend.app.agent.state import RAGAgentState
from backend.app.core.logging import logger


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as numbered sources for the LLM."""
    if not chunks:
        return "No relevant context found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        source_label = (
            f"[Source {i}] "
            f"{chunk.get('doc_name', 'Unknown')} — "
            f"Page {chunk.get('page', '?')}, "
            f"Section: {chunk.get('section', 'N/A')}"
        )
        content = chunk.get("content_markdown") or chunk.get("content", "")
        parts.append(f"{source_label}\n{content}")

    return "\n\n---\n\n".join(parts)


def _extract_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """Extract [Source N] references from the answer and map to chunk metadata."""
    pattern = r"\[Source (\d+)\]"
    referenced = set(int(m) for m in re.findall(pattern, answer))

    citations = []
    for i, chunk in enumerate(chunks, 1):
        if i in referenced:
            citations.append(
                {
                    "source_num": i,
                    "doc_name": chunk.get("doc_name", ""),
                    "page": chunk.get("page", 0),
                    "section": chunk.get("section", ""),
                    "content": chunk.get("content", ""),
                    "score": chunk.get("score", 0.0),
                    "chunk_id": chunk.get("chunk_id", ""),
                }
            )

    return citations


async def generator_node(state: RAGAgentState, config: RunnableConfig = None) -> dict:
    log = logger.bind(node="generator")

    llm = get_chat_model(state["llm_preference"])
    chunks = state.get("reranked_chunks", [])
    context = _build_context(chunks)

    result = await llm.ainvoke(
        [
            SystemMessage(content=GENERATION_SYSTEM),
            HumanMessage(
                content=GENERATION_PROMPT.format(
                    context=context,
                    query=state["original_query"],
                )
            ),
        ],
        config=config,
    )
    answer = result.content

    citations = _extract_citations(answer, chunks)

    log.info(
        "generation_done",
        llm=state["llm_preference"],
        citations=len(citations),
        answer_words=len(answer.split()),
    )

    return {
        "answer": answer,
        "citations": citations,
        "agent_trace": [
            f"Generated answer with {len(citations)} citations "
            f"(llm={state['llm_preference']})"
        ],
    }


async def direct_response(state: RAGAgentState, config: RunnableConfig = None) -> dict:
    """Greeting response — no LLM call needed."""
    return {
        "answer": GREETING_RESPONSE,
        "citations": [],
        "agent_trace": ["Direct response — greeting detected"],
    }


async def direct_llm(state: RAGAgentState, config: RunnableConfig = None) -> dict:
    """General knowledge — LLM without retrieval."""
    llm = get_chat_model(state["llm_preference"])

    result = await llm.ainvoke(
        [
            SystemMessage(content=DIRECT_LLM_SYSTEM),
            HumanMessage(content=state["original_query"]),
        ],
        config=config,
    )

    return {
        "answer": result.content,
        "citations": [],
        "agent_trace": [f"Direct LLM — no retrieval (llm={state['llm_preference']})"],
    }
