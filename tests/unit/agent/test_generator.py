from unittest.mock import AsyncMock, patch, MagicMock

from backend.app.agent.nodes.generator import (
    generator_node,
    direct_response,
    direct_llm,
    _build_context,
    _extract_citations,
)


def _make_state(chunks: list = None, llm_preference: str = "openai") -> dict:
    return {
        "original_query": "What is the revenue?",
        "doc_ids": [],
        "llm_preference": llm_preference,
        "query_type": "factual",
        "sub_questions": [],
        "extracted_filters": {},
        "detected_language": "en",
        "rewritten_query": "What is the revenue?",
        "hyde_query": "",
        "hyde_used": False,
        "retrieved_chunks": [],
        "reranked_chunks": chunks or [],
        "retrieval_attempts": 1,
        "retrieval_quality": 0.8,
        "answer": "",
        "citations": [],
        "agent_trace": [],
        "error": "",
    }


def test_build_context_formats_sources():
    chunks = [
        {"content": "Revenue was $10M", "doc_name": "report.pdf", "page": 5, "section": "Financials"},
        {"content": "Q2 grew 15%", "doc_name": "report.pdf", "page": 8, "section": "Growth"},
    ]
    context = _build_context(chunks)
    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "Revenue was $10M" in context


def test_build_context_empty():
    assert _build_context([]) == "No relevant context found."


def test_extract_citations():
    answer = "Revenue was $10M [Source 1]. Growth was 15% [Source 2]."
    chunks = [
        {"doc_name": "a.pdf", "page": 1, "section": "A", "content": "Rev", "score": 0.9, "chunk_id": "c1"},
        {"doc_name": "b.pdf", "page": 2, "section": "B", "content": "Growth", "score": 0.8, "chunk_id": "c2"},
    ]
    citations = _extract_citations(answer, chunks)
    assert len(citations) == 2
    assert citations[0]["doc_name"] == "a.pdf"


async def test_direct_response_returns_greeting():
    state = _make_state()
    state["query_type"] = "greeting"
    result = await direct_response(state)
    assert result["answer"] != ""
    assert result["citations"] == []


@patch("backend.app.agent.nodes.generator.get_chat_model")
async def test_direct_llm_no_retrieval(mock_get_chat):
    mock_llm = AsyncMock()
    mock_get_chat.return_value = mock_llm
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="The sky is blue."))

    state = _make_state()
    state["query_type"] = "general"
    result = await direct_llm(state)

    assert result["answer"] == "The sky is blue."
    assert result["citations"] == []


@patch("backend.app.agent.nodes.generator.get_chat_model")
async def test_generator_produces_answer(mock_get_chat):
    mock_llm = AsyncMock()
    mock_get_chat.return_value = mock_llm
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(content="Revenue was $10M [Source 1].")
    )

    chunks = [
        {"content": "Revenue was $10M", "doc_name": "report.pdf", "page": 5,
         "section": "Financials", "score": 0.9, "chunk_id": "c1"},
    ]
    state = _make_state(chunks=chunks)
    result = await generator_node(state)

    assert "Revenue" in result["answer"]
    assert len(result["citations"]) >= 1
