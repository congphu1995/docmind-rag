import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_cohere():
    with patch("backend.app.pipeline.rerankers.cohere_reranker.cohere") as mock:
        client = MagicMock()
        mock.ClientV2.return_value = client
        yield client


def _make_chunks(n: int) -> list[dict]:
    return [
        {"content": f"chunk {i}", "score": 1 - i * 0.05, "doc_name": "test.pdf"}
        for i in range(n)
    ]


async def test_cohere_reranker_returns_top_n(mock_cohere):
    from backend.app.pipeline.rerankers.cohere_reranker import CohereReranker

    mock_results = [MagicMock(index=i, relevance_score=1 - i * 0.1) for i in range(3)]
    mock_cohere.rerank.return_value = MagicMock(results=mock_results)

    reranker = CohereReranker(api_key="test-key")
    chunks = _make_chunks(10)
    result = await reranker.rerank("revenue question", chunks, top_n=3)

    assert len(result) == 3
    mock_cohere.rerank.assert_called_once()


async def test_cohere_reranker_preserves_chunk_data(mock_cohere):
    from backend.app.pipeline.rerankers.cohere_reranker import CohereReranker

    mock_results = [MagicMock(index=2, relevance_score=0.95)]
    mock_cohere.rerank.return_value = MagicMock(results=mock_results)

    reranker = CohereReranker(api_key="test-key")
    chunks = _make_chunks(5)
    result = await reranker.rerank("query", chunks, top_n=1)

    assert result[0]["content"] == "chunk 2"
    assert result[0]["score"] == 0.95


async def test_cohere_reranker_falls_back_on_error(mock_cohere):
    from backend.app.pipeline.rerankers.cohere_reranker import CohereReranker

    mock_cohere.rerank.side_effect = Exception("API error")

    reranker = CohereReranker(api_key="test-key")
    chunks = _make_chunks(10)
    result = await reranker.rerank("query", chunks, top_n=5)

    assert len(result) == 5
    assert result[0]["content"] == "chunk 0"
