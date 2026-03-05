import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from backend.app.agent.nodes.retriever import retriever_node, _assess_quality


def _make_state(
    rewritten_query: str = "test query",
    hyde_query: str = "",
    sub_questions: list = None,
) -> dict:
    return {
        "original_query": "test",
        "doc_ids": [],
        "llm_preference": "openai",
        "query_type": "factual",
        "sub_questions": sub_questions or [],
        "extracted_filters": {},
        "detected_language": "en",
        "rewritten_query": rewritten_query,
        "hyde_query": hyde_query,
        "hyde_used": bool(hyde_query),
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "retrieval_attempts": 0,
        "retrieval_quality": 0.0,
        "answer": "",
        "citations": [],
        "agent_trace": [],
        "error": "",
    }


def test_assess_quality_high_scores():
    results = [MagicMock(score=0.9), MagicMock(score=0.85), MagicMock(score=0.8)]
    assert _assess_quality(results) > 0.8


def test_assess_quality_low_scores():
    results = [MagicMock(score=0.3), MagicMock(score=0.2)]
    assert _assess_quality(results) < 0.4


def test_assess_quality_empty():
    assert _assess_quality([]) == 0.0


@patch("backend.app.agent.nodes.retriever._fetch_parents")
@patch("backend.app.agent.nodes.retriever.QdrantWrapper")
@patch("backend.app.agent.nodes.retriever.OpenAIEmbedder")
async def test_retriever_uses_hyde_query(mock_embedder_cls, mock_qdrant_cls, mock_fetch):
    mock_embedder = AsyncMock()
    mock_embedder_cls.return_value = mock_embedder
    mock_embedder.embed_single = AsyncMock(return_value=[0.1] * 1536)

    mock_qdrant = MagicMock()
    mock_qdrant_cls.return_value = mock_qdrant
    mock_qdrant.search = AsyncMock(return_value=[
        MagicMock(score=0.9, payload={"parent_id": "p1", "doc_id": "d1"})
    ])

    mock_fetch.return_value = [{"content": "result", "score": 0.9, "chunk_id": "c1"}]

    state = _make_state(hyde_query="hypothetical answer about revenue")
    result = await retriever_node(state)

    assert len(result["retrieved_chunks"]) > 0
    assert result["retrieval_attempts"] >= 1


@patch("backend.app.agent.nodes.retriever._fetch_parents")
@patch("backend.app.agent.nodes.retriever.QdrantWrapper")
@patch("backend.app.agent.nodes.retriever.OpenAIEmbedder")
async def test_retriever_retries_on_low_quality(mock_embedder_cls, mock_qdrant_cls, mock_fetch):
    mock_embedder = AsyncMock()
    mock_embedder_cls.return_value = mock_embedder
    mock_embedder.embed_single = AsyncMock(return_value=[0.1] * 1536)

    mock_qdrant = MagicMock()
    mock_qdrant_cls.return_value = mock_qdrant
    # First attempt: low scores, second: high scores
    mock_qdrant.search = AsyncMock(side_effect=[
        [MagicMock(score=0.3, payload={"parent_id": "p1", "doc_id": "d1"})],
        [MagicMock(score=0.85, payload={"parent_id": "p1", "doc_id": "d1"})],
    ])

    mock_fetch.return_value = [{"content": "result", "score": 0.85, "chunk_id": "c1"}]

    with patch("backend.app.agent.nodes.retriever.LLMFactory") as mock_llm_factory:
        mock_llm = AsyncMock()
        mock_llm_factory.create_mini.return_value = mock_llm
        mock_llm.complete = AsyncMock(return_value="expanded query")

        state = _make_state()
        result = await retriever_node(state)

        assert result["retrieval_attempts"] == 2
