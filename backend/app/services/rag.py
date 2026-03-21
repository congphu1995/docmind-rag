"""
Orchestrates the RAG agent pipeline.
- query(): Non-streaming — runs full graph, returns complete response.
- stream_query(): Streaming — runs nodes sequentially, streams generation via SSE.
"""

import json
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from backend.app.agent.graph import build_graph
from backend.app.agent.llm import get_chat_model
from backend.app.agent.nodes.decomposer import decomposer
from backend.app.agent.nodes.generator import (
    _build_context,
    direct_llm,
    direct_response,
)
from backend.app.agent.nodes.query_analyzer import query_analyzer
from backend.app.agent.nodes.query_rewriter import query_rewriter
from backend.app.agent.nodes.reranker import reranker_node
from backend.app.agent.nodes.retriever import retriever_node
from backend.app.agent.prompts import GENERATION_PROMPT, GENERATION_SYSTEM
from backend.app.agent.state import RAGAgentState
from backend.app.core.langfuse import get_langfuse_callback
from backend.app.core.logging import logger
from backend.app.schemas.chat import ChatRequest


class RAGService:
    def __init__(self):
        self._graph = build_graph()

    def _build_initial_state(self, request: ChatRequest) -> RAGAgentState:
        return {
            "original_query": request.question,
            "doc_ids": request.doc_ids,
            "llm_preference": request.llm,
            "query_type": "",
            "sub_questions": [],
            "extracted_filters": {},
            "detected_language": "",
            "rewritten_query": "",
            "hyde_query": "",
            "hyde_used": False,
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "retrieval_attempts": 0,
            "retrieval_quality": 0.0,
            "answer": "",
            "citations": [],
            "agent_trace": [],
            "error": "",
        }

    def _build_config(self) -> dict:
        """Build LangGraph config with Langfuse callback if enabled."""
        cb = get_langfuse_callback()
        if cb:
            return {"callbacks": [cb]}
        return {}

    async def query(self, request: ChatRequest) -> dict:
        """Non-streaming: run full graph, return complete response."""
        log = logger.bind(question=request.question[:80])
        log.info("rag_query_start")

        state = self._build_initial_state(request)
        config = self._build_config()

        result = await self._graph.ainvoke(state, config=config)

        log.info(
            "rag_query_done",
            query_type=result.get("query_type"),
            citations=len(result.get("citations", [])),
        )

        return {
            "answer": result.get("answer", ""),
            "sources": result.get("citations", []),
            "reranked_chunks": result.get("reranked_chunks", []),
            "llm_used": request.llm,
            "hyde_used": result.get("hyde_used", False),
            "query_type": result.get("query_type", ""),
            "agent_trace": result.get("agent_trace", []),
        }

    async def stream_query(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """
        Streaming: run nodes sequentially, then stream generation.

        SSE format:
        1. __META__{...}__META__  (sources, trace, metadata)
        2. token by token         (streamed answer)
        3. [DONE]                 (terminal signal)
        """
        log = logger.bind(question=request.question[:80])
        log.info("rag_stream_start")

        state = self._build_initial_state(request)
        config = self._build_config()
        runnable_config = RunnableConfig(**config) if config else None

        # Phase 1: Query analysis
        analyzer_result = await query_analyzer(state, runnable_config)
        state = {**state, **analyzer_result}

        query_type = state["query_type"]

        # Handle non-retrieval paths
        if query_type == "greeting":
            result = await direct_response(state, runnable_config)
            state = {**state, **result}
            yield self._build_meta_event(state, request.llm)
            yield state["answer"]
            yield "[DONE]"
            return

        if query_type == "general":
            result = await direct_llm(state, runnable_config)
            state = {**state, **result}
            yield self._build_meta_event(state, request.llm)
            yield state["answer"]
            yield "[DONE]"
            return

        # Phase 2: Decompose (multi_hop only)
        if query_type == "multi_hop":
            decompose_result = await decomposer(state, runnable_config)
            state = {**state, **decompose_result}

        # Phase 3: Query rewrite + conditional HyDE
        rewrite_result = await query_rewriter(state, runnable_config)
        state = {**state, **rewrite_result}

        # Phase 4: Retrieval with adaptive retry
        retrieval_result = await retriever_node(state, runnable_config)
        state = {**state, **retrieval_result}

        # Phase 5: Rerank
        rerank_result = await reranker_node(state)
        state = {**state, **rerank_result}

        # Yield META event (sources + trace before generation)
        yield self._build_meta_event(state, request.llm)

        # Phase 6: Stream generation
        llm = get_chat_model(request.llm, streaming=True).bind(
            temperature=0.1, max_tokens=4096
        )
        context = _build_context(state.get("reranked_chunks", []))
        messages = [
            SystemMessage(content=GENERATION_SYSTEM),
            HumanMessage(
                content=GENERATION_PROMPT.format(
                    context=context,
                    query=state["original_query"],
                )
            ),
        ]

        full_answer = ""
        async for chunk in llm.astream(messages, config=runnable_config):
            if chunk.content:
                full_answer += chunk.content
                yield chunk.content

        yield "[DONE]"

        log.info(
            "rag_stream_done",
            query_type=query_type,
            answer_words=len(full_answer.split()),
        )

    def _build_meta_event(self, state: dict, llm_used: str) -> str:
        """Build __META__ SSE event with sources and agent trace."""
        sources = []
        for i, chunk in enumerate(state.get("reranked_chunks", []), 1):
            sources.append(
                {
                    "source_num": i,
                    "doc_name": chunk.get("doc_name", ""),
                    "page": chunk.get("page", 0),
                    "section": chunk.get("section", ""),
                    "content": chunk.get("content", ""),
                    "score": chunk.get("score", 0.0),
                }
            )

        meta = {
            "sources": sources,
            "llm_used": llm_used,
            "hyde_used": state.get("hyde_used", False),
            "query_type": state.get("query_type", ""),
            "agent_trace": state.get("agent_trace", []),
        }

        return f"__META__{json.dumps(meta)}__META__"
