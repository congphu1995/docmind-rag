from unittest.mock import patch, AsyncMock

from backend.app.agent.nodes.reranker import reranker_node


def _make_state(chunks: list = None) -> dict:
    return {
        "original_query": "What is the revenue?",
        "doc_ids": [],
        "llm_preference": "openai",
        "query_type": "factual",
        "sub_questions": [],
        "extracted_filters": {},
        "detected_language": "en",
        "rewritten_query": "",
        "hyde_query": "",
        "hyde_used": False,
        "retrieved_chunks": chunks or [],
        "reranked_chunks": [],
        "retrieval_attempts": 1,
        "retrieval_quality": 0.8,
        "answer": "",
        "citations": [],
        "agent_trace": [],
        "error": "",
    }


async def test_identity_reranker_limits_to_top_n():
    chunks = [{"content": f"chunk {i}", "score": 1 - i * 0.1} for i in range(10)]
    state = _make_state(chunks=chunks)
    result = await reranker_node(state)
    assert len(result["reranked_chunks"]) == 5  # default top_n


async def test_identity_reranker_preserves_order():
    chunks = [
        {"content": "first", "score": 0.9},
        {"content": "second", "score": 0.8},
    ]
    state = _make_state(chunks=chunks)
    result = await reranker_node(state)
    assert result["reranked_chunks"][0]["content"] == "first"


async def test_reranker_handles_empty_chunks():
    state = _make_state(chunks=[])
    result = await reranker_node(state)
    assert result["reranked_chunks"] == []


async def test_reranker_node_uses_configured_strategy():
    chunks = [{"content": f"chunk {i}", "score": 0.9} for i in range(10)]
    state = _make_state(chunks=chunks)

    mock_reranker = AsyncMock()
    mock_reranker.rerank.return_value = chunks[:3]

    with patch("backend.app.agent.nodes.reranker.RerankerFactory") as mock_factory:
        mock_factory.create.return_value = mock_reranker
        await reranker_node(state)
        mock_factory.create.assert_called_once()
