import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.pipeline.embedders.openai_embedder import OpenAIEmbedder


@pytest.fixture
def mock_openai():
    with patch("backend.app.pipeline.embedders.openai_embedder.AsyncOpenAI") as mock:
        client = AsyncMock()
        mock.return_value = client

        embedding_item = MagicMock()
        embedding_item.embedding = [0.1] * 1536

        response = MagicMock()
        response.data = [embedding_item]
        client.embeddings.create = AsyncMock(return_value=response)

        yield client


async def test_embed_single(mock_openai):
    embedder = OpenAIEmbedder()
    result = await embedder.embed_single("test text")
    assert len(result) == 1536


async def test_embed_batch(mock_openai):
    embedder = OpenAIEmbedder()
    # Mock returns one item per call, but we test the batch logic
    embedding_items = [MagicMock(embedding=[0.1] * 1536) for _ in range(3)]
    response = MagicMock()
    response.data = embedding_items
    mock_openai.embeddings.create = AsyncMock(return_value=response)

    result = await embedder.embed(["text1", "text2", "text3"])
    assert len(result) == 3
    assert all(len(v) == 1536 for v in result)


async def test_embed_empty_list(mock_openai):
    embedder = OpenAIEmbedder()
    result = await embedder.embed([])
    assert result == []


def test_dimensions():
    with patch("backend.app.pipeline.embedders.openai_embedder.AsyncOpenAI"):
        embedder = OpenAIEmbedder()
        assert embedder.dimensions == 1536
