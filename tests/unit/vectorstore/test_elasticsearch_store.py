from unittest.mock import AsyncMock, patch

import pytest

from backend.app.pipeline.base.chunker import Chunk
from backend.app.vectorstore.elasticsearch_store import ElasticsearchStore


@pytest.fixture
def mock_es_client():
    client = AsyncMock()
    client.indices = AsyncMock()
    client.indices.exists = AsyncMock(return_value=False)
    client.indices.create = AsyncMock()
    return client


@pytest.fixture
def store(mock_es_client):
    with patch(
        "backend.app.vectorstore.elasticsearch_store.AsyncElasticsearch",
        return_value=mock_es_client,
    ):
        s = ElasticsearchStore()
        s._client = mock_es_client
        return s


async def test_initialize_creates_index(store, mock_es_client):
    mock_es_client.indices.exists.return_value = False
    await store.initialize()
    mock_es_client.indices.create.assert_called_once()
    call_kwargs = mock_es_client.indices.create.call_args
    assert "docmind_chunks" in str(call_kwargs)


async def test_initialize_skips_existing_index(store, mock_es_client):
    mock_es_client.indices.exists.return_value = True
    await store.initialize()
    mock_es_client.indices.create.assert_not_called()


async def test_upsert_chunks_with_vectors(store, mock_es_client):
    mock_es_client.bulk = AsyncMock(return_value={"errors": False, "items": []})
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", content="enriched", content_raw="raw"),
        Chunk(chunk_id="c2", doc_id="d1", content="enriched2", content_raw="raw2"),
    ]
    vectors = [[0.1] * 1536, [0.2] * 1536]
    await store.upsert_chunks(chunks, vectors)
    mock_es_client.bulk.assert_called_once()


async def test_upsert_chunks_without_vectors(store, mock_es_client):
    """Parents are upserted without vectors."""
    mock_es_client.bulk = AsyncMock(return_value={"errors": False, "items": []})
    parents = [
        Chunk(chunk_id="p1", doc_id="d1", content_raw="raw", is_parent=True),
    ]
    await store.upsert_chunks(parents, vectors=None)
    mock_es_client.bulk.assert_called_once()
    call_args = mock_es_client.bulk.call_args
    operations = (
        call_args[1]["operations"] if "operations" in call_args[1] else call_args[0][0]
    )
    docs = [operations[i] for i in range(1, len(operations), 2)]
    for doc in docs:
        assert "embedding" not in doc


async def test_search_returns_merged_rrf_results(store, mock_es_client):
    """search() runs BM25 + kNN and merges with RRF."""
    bm25_response = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "chunk_id": "c1",
                        "parent_id": "p1",
                        "doc_id": "d1",
                        "doc_name": "test.pdf",
                        "content_raw": "text1",
                        "type": "text",
                        "page": 1,
                        "section": "s1",
                        "language": "en",
                        "word_count": 50,
                        "metadata": {},
                    },
                    "_score": 5.0,
                },
                {
                    "_source": {
                        "chunk_id": "c2",
                        "parent_id": "p1",
                        "doc_id": "d1",
                        "doc_name": "test.pdf",
                        "content_raw": "text2",
                        "type": "text",
                        "page": 1,
                        "section": "s1",
                        "language": "en",
                        "word_count": 60,
                        "metadata": {},
                    },
                    "_score": 3.0,
                },
            ]
        }
    }
    knn_response = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "chunk_id": "c2",
                        "parent_id": "p1",
                        "doc_id": "d1",
                        "doc_name": "test.pdf",
                        "content_raw": "text2",
                        "type": "text",
                        "page": 1,
                        "section": "s1",
                        "language": "en",
                        "word_count": 60,
                        "metadata": {},
                    },
                    "_score": 0.95,
                },
                {
                    "_source": {
                        "chunk_id": "c3",
                        "parent_id": "p2",
                        "doc_id": "d1",
                        "doc_name": "test.pdf",
                        "content_raw": "text3",
                        "type": "text",
                        "page": 2,
                        "section": "s2",
                        "language": "en",
                        "word_count": 70,
                        "metadata": {},
                    },
                    "_score": 0.8,
                },
            ]
        }
    }
    mock_es_client.search = AsyncMock(side_effect=[bm25_response, knn_response])

    results = await store.search(
        query_vector=[0.1] * 1536,
        query_text="test query",
        top_k=3,
    )

    assert len(results) <= 3
    assert results[0]["chunk_id"] == "c2"
    assert "score" in results[0]
    assert isinstance(results[0]["score"], float)


async def test_rrf_merge_basic(store):
    bm25 = [
        {"chunk_id": "a", "content_raw": "x"},
        {"chunk_id": "b", "content_raw": "y"},
    ]
    knn = [
        {"chunk_id": "b", "content_raw": "y"},
        {"chunk_id": "c", "content_raw": "z"},
    ]
    merged = store._rrf_merge(bm25, knn, k=60)
    ids = [r["chunk_id"] for r in merged]
    assert ids[0] == "b"
    assert len(merged) == 3


async def test_fetch_parents(store, mock_es_client):
    mock_es_client.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "chunk_id": "p1",
                            "content_raw": "parent text",
                            "content_markdown": "**parent**",
                            "is_parent": True,
                            "doc_id": "d1",
                            "doc_name": "test.pdf",
                            "page": 1,
                            "section": "s1",
                            "type": "text",
                            "language": "en",
                            "word_count": 200,
                            "metadata": {},
                        }
                    },
                ]
            }
        }
    )
    parents = await store.fetch_parents(["p1"])
    assert len(parents) == 1
    assert parents[0]["chunk_id"] == "p1"
    assert parents[0]["content_raw"] == "parent text"


async def test_delete_by_doc_id(store, mock_es_client):
    mock_es_client.delete_by_query = AsyncMock(return_value={"deleted": 5})
    await store.delete_by_doc_id("d1")
    mock_es_client.delete_by_query.assert_called_once()


async def test_get_by_doc_id(store, mock_es_client):
    mock_es_client.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "chunk_id": "c1",
                            "is_parent": False,
                            "doc_id": "d1",
                            "content_raw": "child",
                            "metadata": {},
                        }
                    },
                    {
                        "_source": {
                            "chunk_id": "p1",
                            "is_parent": True,
                            "doc_id": "d1",
                            "content_raw": "parent",
                            "metadata": {},
                        }
                    },
                ]
            }
        }
    )
    chunks = await store.get_by_doc_id("d1")
    assert len(chunks) == 2


def test_build_filters_default(store):
    filters = store._build_filters(None)
    assert {"term": {"is_parent": False}} in filters
    assert len(filters) == 1


def test_build_filters_with_doc_ids(store):
    filters = store._build_filters({"doc_ids": ["d1", "d2"]})
    assert {"term": {"is_parent": False}} in filters
    assert {"terms": {"doc_id": ["d1", "d2"]}} in filters


def test_build_filters_with_all_options(store):
    filters = store._build_filters(
        {
            "doc_ids": ["d1"],
            "language": "en",
            "type": "text",
            "user_id": "u1",
        }
    )
    assert len(filters) == 5


async def test_score_threshold_is_accepted_but_not_applied(store, mock_es_client):
    bm25_resp = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "chunk_id": "c1",
                        "parent_id": "p1",
                        "doc_id": "d1",
                        "doc_name": "t.pdf",
                        "content_raw": "x",
                        "type": "text",
                        "page": 1,
                        "section": "s",
                        "language": "en",
                        "word_count": 10,
                        "metadata": {},
                    },
                    "_score": 1.0,
                },
            ]
        }
    }
    knn_resp = {"hits": {"hits": []}}
    mock_es_client.search = AsyncMock(side_effect=[bm25_resp, knn_resp])

    results = await store.search(
        query_vector=[0.1] * 1536,
        query_text="test",
        top_k=5,
        score_threshold=0.99,
    )
    assert len(results) == 1
