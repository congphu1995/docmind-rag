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
    results = [{"score": 0.9}, {"score": 0.85}, {"score": 0.8}]
    assert _assess_quality(results) > 0.8


def test_assess_quality_low_scores():
    results = [{"score": 0.3}, {"score": 0.2}]
    assert _assess_quality(results) < 0.4


def test_assess_quality_empty():
    assert _assess_quality([]) == 0.0


@patch("backend.app.agent.nodes.retriever._fetch_parents")
@patch("backend.app.agent.nodes.retriever.VectorStoreFactory")
@patch("backend.app.agent.nodes.retriever.OpenAIEmbedder")
async def test_retriever_uses_hyde_query(
    mock_embedder_cls, mock_factory_cls, mock_fetch
):
    mock_embedder = AsyncMock()
    mock_embedder_cls.return_value = mock_embedder
    mock_embedder.embed_single = AsyncMock(return_value=[0.1] * 1536)

    mock_store = AsyncMock()
    mock_factory_cls.create.return_value = mock_store
    mock_store.search = AsyncMock(
        return_value=[
            {
                "score": 0.9,
                "parent_id": "p1",
                "doc_id": "d1",
                "chunk_id": "c1",
                "content_raw": "text",
                "doc_name": "test.pdf",
                "type": "text",
                "page": 1,
                "section": "s1",
                "language": "en",
                "word_count": 50,
            }
        ]
    )

    mock_fetch.return_value = [{"content": "result", "score": 0.9, "chunk_id": "c1"}]

    state = _make_state(hyde_query="hypothetical answer about revenue")
    result = await retriever_node(state)

    assert len(result["retrieved_chunks"]) > 0
    assert result["retrieval_attempts"] >= 1


@patch("backend.app.agent.nodes.retriever._fetch_parents")
@patch("backend.app.agent.nodes.retriever.VectorStoreFactory")
@patch("backend.app.agent.nodes.retriever.OpenAIEmbedder")
async def test_retriever_retries_on_low_quality(
    mock_embedder_cls, mock_factory_cls, mock_fetch
):
    mock_embedder = AsyncMock()
    mock_embedder_cls.return_value = mock_embedder
    mock_embedder.embed_single = AsyncMock(return_value=[0.1] * 1536)

    mock_store = AsyncMock()
    mock_factory_cls.create.return_value = mock_store
    mock_store.search = AsyncMock(
        side_effect=[
            [
                {
                    "score": 0.3,
                    "parent_id": "p1",
                    "doc_id": "d1",
                    "chunk_id": "c1",
                    "content_raw": "text",
                    "doc_name": "test.pdf",
                    "type": "text",
                    "page": 1,
                    "section": "s1",
                    "language": "en",
                    "word_count": 50,
                }
            ],
            [
                {
                    "score": 0.85,
                    "parent_id": "p1",
                    "doc_id": "d1",
                    "chunk_id": "c1",
                    "content_raw": "text",
                    "doc_name": "test.pdf",
                    "type": "text",
                    "page": 1,
                    "section": "s1",
                    "language": "en",
                    "word_count": 50,
                }
            ],
        ]
    )

    mock_fetch.return_value = [{"content": "result", "score": 0.85, "chunk_id": "c1"}]

    with patch("backend.app.agent.nodes.retriever.get_mini_model") as mock_get_mini:
        mock_llm = MagicMock()
        mock_get_mini.return_value = mock_llm
        mock_bound = AsyncMock()
        mock_llm.bind.return_value = mock_bound
        mock_bound.ainvoke = AsyncMock(return_value=MagicMock(content="expanded query"))

        state = _make_state()
        result = await retriever_node(state)

        assert result["retrieval_attempts"] == 2
