"""
LangGraph typed state — flows through every agent node.
Uses Annotated with operator.add for agent_trace accumulation.
"""

import operator
from typing import Annotated, TypedDict


class RAGAgentState(TypedDict):
    # ── Input ────────────────────────────────────────────────
    original_query: str
    doc_ids: list[str]
    llm_preference: str  # "claude" | "openai"

    # ── Query Understanding ──────────────────────────────────
    query_type: str  # factual|analytical|tabular|multi_hop|general|greeting
    sub_questions: list[str]
    extracted_filters: dict
    detected_language: str

    # ── Query Rewriting ──────────────────────────────────────
    rewritten_query: str
    hyde_query: str  # empty string if HyDE skipped
    hyde_used: bool

    # ── Retrieval ────────────────────────────────────────────
    retrieved_chunks: list[dict]
    reranked_chunks: list[dict]
    retrieval_attempts: int
    retrieval_quality: float

    # ── Generation ───────────────────────────────────────────
    answer: str
    citations: list[dict]

    # ── Observability ────────────────────────────────────────
    agent_trace: Annotated[list[str], operator.add]  # accumulates across nodes
    error: str
