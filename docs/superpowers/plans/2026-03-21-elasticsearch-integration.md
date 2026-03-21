# Elasticsearch Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Qdrant + PostgreSQL parent chunk storage with Elasticsearch, enabling hybrid search (dense vectors + BM25 via application-side RRF).

**Architecture:** Single ES index stores both parent and child chunks. `BaseVectorStore` ABC + `ElasticsearchStore` implementation + `VectorStoreFactory` — following the project's existing strategy pattern. Two-query RRF fusion at the application layer guarantees ES free Basic license compatibility.

**Tech Stack:** `elasticsearch[async]` (AsyncElasticsearch), Elasticsearch 8.15.0, Python 3.11+

**Spec:** `docs/superpowers/specs/2026-03-21-elasticsearch-integration-design.md`

---

## File Map

### New Files
| File | Responsibility |
|---|---|
| `backend/app/pipeline/base/vectorstore.py` | BaseVectorStore ABC — 6 abstract methods |
| `backend/app/vectorstore/elasticsearch_store.py` | ElasticsearchStore — hybrid search, RRF, bulk upsert |
| `backend/app/vectorstore/factory.py` | VectorStoreFactory — config-driven instantiation |
| `tests/unit/vectorstore/test_elasticsearch_store.py` | Unit tests for ElasticsearchStore (mocked ES client) |
| `tests/unit/vectorstore/__init__.py` | Package init |
| `tests/unit/vectorstore/test_factory.py` | Unit tests for VectorStoreFactory |

### Modified Files
| File | Changes |
|---|---|
| `backend/app/pipeline/base/chunker.py` | Add `user_id` field, rename `qdrant_payload()` → `to_document()` |
| `backend/app/core/config.py` | Add ES settings (url, index, auth, rrf_k, batch_size) |
| `backend/app/services/ingestion.py` | Use vectorstore for parents+children, update `delete_document()` |
| `backend/app/agent/nodes/retriever.py` | Use vectorstore.search()+fetch_parents(), fix dict access |
| `backend/app/api/chunks.py` | Use vectorstore.get_by_doc_id(), fix ORM→dict access |
| `backend/app/main.py` | Initialize vectorstore in lifespan |
| `backend/app/models/document.py` | Remove `ParentChunk` class |
| `docker-compose.yml` | Add ES service, remove Qdrant service |
| `pyproject.toml` | Add `elasticsearch[async]`, remove `qdrant-client` |
| `.env.example` | Add ES vars, remove Qdrant vars |
| `tests/unit/pipeline/test_schemas.py` | Update qdrant_payload tests → to_document |
| `tests/unit/agent/test_retriever.py` | Update mocks from QdrantWrapper → vectorstore |
| `tests/integration/test_ingestion_pipeline.py` | Update Qdrant refs → vectorstore |
| `tests/integration/test_chat_pipeline.py` | Update docstring |
| `scripts/seed_demo_data.py` | Update docstring |
| `scripts/seed_custom_eval.py` | Update docstring |
| `CLAUDE.md` | Update vectorstore references |

### Deleted Files
| File | Reason |
|---|---|
| `backend/app/vectorstore/qdrant_client.py` | Replaced by elasticsearch_store.py |

---

## Task 1: Infrastructure Setup (dependencies, config, docker)

**Files:**
- Modify: `pyproject.toml`
- Modify: `backend/app/core/config.py`
- Modify: `.env.example`
- Modify: `docker-compose.yml`

- [ ] **Step 1: Update Python dependencies**

In `pyproject.toml`, replace `qdrant-client` with `elasticsearch[async]`:

```toml
# Remove this line:
    "qdrant-client>=1.17.0",
# Add this line:
    "elasticsearch[async]>=8.15.0",
```

- [ ] **Step 2: Install dependencies**

Run: `uv sync`
Expected: resolves successfully with elasticsearch package

- [ ] **Step 3: Add ES settings to config**

In `backend/app/core/config.py`, add after the `# Qdrant` section:

```python
    # Elasticsearch
    vectorstore_strategy: str = "elasticsearch"
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "docmind_chunks"
    elasticsearch_username: str = ""
    elasticsearch_password: str = ""
    rrf_k: int = 60
    es_bulk_batch_size: int = 100
```

Keep the existing Qdrant settings (for backward compat), just add the new ones.

- [ ] **Step 4: Update .env.example**

Replace Qdrant vars with ES vars:

```env
# Remove:
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Add:
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX=docmind_chunks
```

- [ ] **Step 5: Update docker-compose.yml**

Replace the `qdrant` service (lines 2-10) with:

```yaml
  elasticsearch:
    image: elasticsearch:8.15.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - xpack.security.http.ssl.enabled=false
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
      - TZ=Asia/Ho_Chi_Minh
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:9200/_cluster/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 10
```

Update `backend` service (lines 153-169):
- Replace `- QDRANT_HOST=qdrant` with `- ELASTICSEARCH_URL=http://elasticsearch:9200`
- Replace `depends_on` `- qdrant` with:
```yaml
      elasticsearch:
        condition: service_healthy
```

Update `worker` service (lines 171-186):
- Same changes as backend: replace `QDRANT_HOST` with `ELASTICSEARCH_URL`, update `depends_on`

Update `volumes` (line 218-225):
- Replace `qdrant_data:` with `elasticsearch_data:`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml backend/app/core/config.py .env.example docker-compose.yml
git commit -m "feat: add Elasticsearch infrastructure, config, and dependencies"
```

---

## Task 2: Update Chunk Dataclass

**Files:**
- Modify: `backend/app/pipeline/base/chunker.py`
- Test: `tests/unit/pipeline/test_schemas.py`

- [ ] **Step 1: Write tests for the new `to_document()` method**

Replace the two `qdrant_payload` tests in `tests/unit/pipeline/test_schemas.py` (lines 60-78) with:

```python
def test_chunk_to_document_includes_all_fields():
    chunk = Chunk(
        doc_id="abc",
        content="enriched text",
        content_raw="raw text",
        content_html="<p>raw text</p>",
        page=1,
        is_parent=False,
        user_id="user1",
        metadata={"doc_type": "report"},
    )
    doc = chunk.to_document()
    assert isinstance(doc, dict)
    assert doc["doc_id"] == "abc"
    assert doc["content"] == "enriched text"
    assert doc["content_raw"] == "raw text"
    assert doc["content_html"] == "<p>raw text</p>"
    assert doc["user_id"] == "user1"
    assert doc["is_parent"] is False
    assert doc["metadata"] == {"doc_type": "report"}
    assert "created_at" in doc


def test_chunk_to_document_metadata_is_nested():
    """Metadata must be a nested dict, not flattened into top-level keys."""
    chunk = Chunk(
        doc_id="abc",
        content="test",
        content_raw="test",
        metadata={"doc_type": "report", "date": "2024-01-01"},
    )
    doc = chunk.to_document()
    # metadata is nested, not flattened
    assert "metadata" in doc
    assert doc["metadata"]["doc_type"] == "report"
    # top-level should NOT have doc_type
    assert "doc_type" not in doc


def test_chunk_default_user_id():
    chunk = Chunk()
    assert chunk.user_id == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/pipeline/test_schemas.py::test_chunk_to_document_includes_all_fields tests/unit/pipeline/test_schemas.py::test_chunk_to_document_metadata_is_nested tests/unit/pipeline/test_schemas.py::test_chunk_default_user_id -v`
Expected: FAIL — `to_document` and `user_id` don't exist yet

- [ ] **Step 3: Implement Chunk changes**

In `backend/app/pipeline/base/chunker.py`:

Add `from datetime import datetime, timezone` at top.

Add `user_id: str = ""` field after `doc_name` (line 14, after `doc_name: str = ""`):

```python
    user_id: str = ""
```

Replace `qdrant_payload()` method (lines 30-45) with:

```python
    def to_document(self) -> dict:
        """Serialize chunk to a dict suitable for any vector store."""
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "user_id": self.user_id,
            "content": self.content,
            "content_raw": self.content_raw,
            "content_markdown": self.content_markdown,
            "content_html": self.content_html,
            "type": self.type,
            "page": self.page,
            "section": self.section,
            "language": self.language,
            "word_count": self.word_count,
            "is_parent": self.is_parent,
            "metadata": self.metadata,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
```

Update the `BaseChunker` docstring (lines 55-59) to remove PostgreSQL/Qdrant references:

```python
        """
        Returns (parent_chunks, child_chunks).
        parent_chunks → stored for LLM context
        child_chunks  → embedded for retrieval
        """
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/pipeline/test_schemas.py -v`
Expected: ALL PASS (including existing tests like `test_chunk_default_values`)

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/base/chunker.py tests/unit/pipeline/test_schemas.py
git commit -m "feat: add user_id to Chunk, rename qdrant_payload to to_document"
```

---

## Task 3: BaseVectorStore ABC

**Files:**
- Create: `backend/app/pipeline/base/vectorstore.py`

- [ ] **Step 1: Create the ABC**

```python
from abc import ABC, abstractmethod
from typing import Optional

from backend.app.pipeline.base.chunker import Chunk


class BaseVectorStore(ABC):

    @abstractmethod
    async def initialize(self) -> None:
        """Create index/collection if not exists."""
        ...

    @abstractmethod
    async def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]] | None = None,
    ) -> None:
        """Upsert chunks. Vectors provided for children, None for parents."""
        ...

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int = 20,
        filters: Optional[dict] = None,
        score_threshold: Optional[float] = None,
    ) -> list[dict]:
        """
        Hybrid search for similar child chunks.
        Returns list of dicts with keys: score, chunk_id, parent_id,
        doc_id, doc_name, content_raw, type, page, section, language,
        word_count, plus metadata fields.
        """
        ...

    @abstractmethod
    async def fetch_parents(
        self,
        parent_ids: list[str],
    ) -> list[dict]:
        """Fetch parent chunks. parent_ids are chunk_id values of parent documents."""
        ...

    @abstractmethod
    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all chunks (parents + children) for a document."""
        ...

    @abstractmethod
    async def get_by_doc_id(self, doc_id: str) -> list[dict]:
        """Get all chunks for a document (chunk viewer)."""
        ...
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from backend.app.pipeline.base.vectorstore import BaseVectorStore; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/app/pipeline/base/vectorstore.py
git commit -m "feat: add BaseVectorStore ABC"
```

---

## Task 4: ElasticsearchStore Implementation

**Files:**
- Create: `backend/app/vectorstore/elasticsearch_store.py`
- Create: `tests/unit/vectorstore/__init__.py`
- Create: `tests/unit/vectorstore/test_elasticsearch_store.py`

- [ ] **Step 1: Write unit tests for ElasticsearchStore**

Create `tests/unit/vectorstore/__init__.py` (empty file).

Create `tests/unit/vectorstore/test_elasticsearch_store.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch

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
    # Verify no embedding field in the bulk body
    call_args = mock_es_client.bulk.call_args
    operations = call_args[1]["operations"] if "operations" in call_args[1] else call_args[0][0]
    # Find the document body (every other item after the action)
    docs = [operations[i] for i in range(1, len(operations), 2)]
    for doc in docs:
        assert "embedding" not in doc


async def test_search_returns_merged_rrf_results(store, mock_es_client):
    """search() runs BM25 + kNN and merges with RRF."""
    bm25_response = {
        "hits": {"hits": [
            {"_source": {"chunk_id": "c1", "parent_id": "p1", "doc_id": "d1",
                         "doc_name": "test.pdf", "content_raw": "text1",
                         "type": "text", "page": 1, "section": "s1",
                         "language": "en", "word_count": 50, "metadata": {}},
             "_score": 5.0},
            {"_source": {"chunk_id": "c2", "parent_id": "p1", "doc_id": "d1",
                         "doc_name": "test.pdf", "content_raw": "text2",
                         "type": "text", "page": 1, "section": "s1",
                         "language": "en", "word_count": 60, "metadata": {}},
             "_score": 3.0},
        ]}
    }
    knn_response = {
        "hits": {"hits": [
            {"_source": {"chunk_id": "c2", "parent_id": "p1", "doc_id": "d1",
                         "doc_name": "test.pdf", "content_raw": "text2",
                         "type": "text", "page": 1, "section": "s1",
                         "language": "en", "word_count": 60, "metadata": {}},
             "_score": 0.95},
            {"_source": {"chunk_id": "c3", "parent_id": "p2", "doc_id": "d1",
                         "doc_name": "test.pdf", "content_raw": "text3",
                         "type": "text", "page": 2, "section": "s2",
                         "language": "en", "word_count": 70, "metadata": {}},
             "_score": 0.8},
        ]}
    }
    mock_es_client.search = AsyncMock(side_effect=[bm25_response, knn_response])

    results = await store.search(
        query_vector=[0.1] * 1536,
        query_text="test query",
        top_k=3,
    )

    assert len(results) <= 3
    # c2 appears in BOTH lists → highest RRF score
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
    # "b" appears in both, should be ranked first
    assert ids[0] == "b"
    assert len(merged) == 3  # a, b, c


async def test_fetch_parents(store, mock_es_client):
    mock_es_client.search = AsyncMock(return_value={
        "hits": {"hits": [
            {"_source": {"chunk_id": "p1", "content_raw": "parent text",
                         "content_markdown": "**parent**", "is_parent": True,
                         "doc_id": "d1", "doc_name": "test.pdf",
                         "page": 1, "section": "s1", "type": "text",
                         "language": "en", "word_count": 200, "metadata": {}}},
        ]}
    })
    parents = await store.fetch_parents(["p1"])
    assert len(parents) == 1
    assert parents[0]["chunk_id"] == "p1"
    assert parents[0]["content_raw"] == "parent text"


async def test_delete_by_doc_id(store, mock_es_client):
    mock_es_client.delete_by_query = AsyncMock(return_value={"deleted": 5})
    await store.delete_by_doc_id("d1")
    mock_es_client.delete_by_query.assert_called_once()


async def test_get_by_doc_id(store, mock_es_client):
    mock_es_client.search = AsyncMock(return_value={
        "hits": {"hits": [
            {"_source": {"chunk_id": "c1", "is_parent": False, "doc_id": "d1",
                         "content_raw": "child", "metadata": {}}},
            {"_source": {"chunk_id": "p1", "is_parent": True, "doc_id": "d1",
                         "content_raw": "parent", "metadata": {}}},
        ]}
    })
    chunks = await store.get_by_doc_id("d1")
    assert len(chunks) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/vectorstore/test_elasticsearch_store.py -v`
Expected: FAIL — `elasticsearch_store` module doesn't exist

- [ ] **Step 3: Implement ElasticsearchStore**

Create `backend/app/vectorstore/elasticsearch_store.py`:

```python
"""
Elasticsearch vector store — hybrid search (dense + BM25) with RRF fusion.
"""

from elasticsearch import AsyncElasticsearch

from backend.app.core.config import settings
from backend.app.core.exceptions import VectorStoreError
from backend.app.core.logging import logger
from backend.app.pipeline.base.chunker import Chunk
from backend.app.pipeline.base.vectorstore import BaseVectorStore

INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "parent_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "doc_name": {"type": "keyword"},
            "user_id": {"type": "keyword"},
            "content": {"type": "text", "index": False},
            "content_raw": {"type": "text", "analyzer": "standard"},
            "content_markdown": {"type": "text", "index": False},
            "content_html": {"type": "text", "index": False},
            "embedding": {
                "type": "dense_vector",
                "dims": settings.embedding_dimensions,
                "index": True,
                "similarity": "cosine",
            },
            "type": {"type": "keyword"},
            "page": {"type": "integer"},
            "section": {"type": "keyword"},
            "language": {"type": "keyword"},
            "word_count": {"type": "integer"},
            "is_parent": {"type": "boolean"},
            "metadata": {"type": "object", "dynamic": True},
            "created_at": {"type": "date"},
        }
    },
}


class ElasticsearchStore(BaseVectorStore):

    def __init__(self):
        kwargs = {"hosts": [settings.elasticsearch_url]}
        if settings.elasticsearch_username:
            kwargs["basic_auth"] = (
                settings.elasticsearch_username,
                settings.elasticsearch_password,
            )
        self._client = AsyncElasticsearch(**kwargs)
        self._index = settings.elasticsearch_index
        self._rrf_k = settings.rrf_k
        self._batch_size = settings.es_bulk_batch_size

    async def initialize(self) -> None:
        try:
            exists = await self._client.indices.exists(index=self._index)
            if not exists:
                await self._client.indices.create(
                    index=self._index, body=INDEX_MAPPING
                )
                logger.info("es_index_created", index=self._index)
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ES index: {e}") from e

    async def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]] | None = None,
    ) -> None:
        if not chunks:
            return

        operations = []
        for i, chunk in enumerate(chunks):
            doc = chunk.to_document()
            if vectors is not None and i < len(vectors):
                doc["embedding"] = vectors[i]
            operations.append({"index": {"_index": self._index, "_id": doc["chunk_id"]}})
            operations.append(doc)

        try:
            for batch_start in range(0, len(operations), self._batch_size * 2):
                batch = operations[batch_start : batch_start + self._batch_size * 2]
                response = await self._client.bulk(operations=batch)
                if response.get("errors"):
                    failed = sum(
                        1
                        for item in response["items"]
                        if "error" in item.get("index", {})
                    )
                    raise VectorStoreError(
                        f"ES bulk upsert: {failed}/{len(response['items'])} failed"
                    )
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(f"ES upsert failed: {e}") from e

        logger.info("es_upsert_done", count=len(chunks))

    async def search(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int = 20,
        filters: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        candidate_k = top_k * 3

        try:
            bm25_results = await self._bm25_search(query_text, candidate_k, filters)
            knn_results = await self._knn_search(query_vector, candidate_k, filters)
        except Exception as e:
            raise VectorStoreError(f"ES search failed: {e}") from e

        merged = self._rrf_merge(bm25_results, knn_results, k=self._rrf_k)
        return merged[:top_k]

    async def fetch_parents(self, parent_ids: list[str]) -> list[dict]:
        if not parent_ids:
            return []

        try:
            response = await self._client.search(
                index=self._index,
                body={
                    "query": {
                        "bool": {
                            "filter": [
                                {"terms": {"chunk_id": parent_ids}},
                                {"term": {"is_parent": True}},
                            ]
                        }
                    },
                    "size": len(parent_ids),
                },
            )
        except Exception as e:
            raise VectorStoreError(f"ES fetch_parents failed: {e}") from e

        return [hit["_source"] for hit in response["hits"]["hits"]]

    async def delete_by_doc_id(self, doc_id: str) -> None:
        try:
            await self._client.delete_by_query(
                index=self._index,
                body={"query": {"term": {"doc_id": doc_id}}},
            )
            logger.info("es_delete_done", doc_id=doc_id)
        except Exception as e:
            raise VectorStoreError(f"ES delete failed: {e}") from e

    async def get_by_doc_id(self, doc_id: str) -> list[dict]:
        try:
            response = await self._client.search(
                index=self._index,
                body={
                    "query": {"term": {"doc_id": doc_id}},
                    "size": 10000,
                },
            )
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.warning("es_get_by_doc_failed", error=str(e))
            return []

    # ── Private helpers ──────────────────────────────────────────

    async def _bm25_search(
        self, query_text: str, top_k: int, filters: dict | None
    ) -> list[dict]:
        body: dict = {
            "query": {
                "bool": {
                    "must": [{"match": {"content_raw": query_text}}],
                    "filter": self._build_filters(filters),
                }
            },
            "size": top_k,
        }

        response = await self._client.search(index=self._index, body=body)
        return [hit["_source"] for hit in response["hits"]["hits"]]

    async def _knn_search(
        self, query_vector: list[float], top_k: int, filters: dict | None
    ) -> list[dict]:
        body: dict = {
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": top_k * 5,
                "filter": {"bool": {"filter": self._build_filters(filters)}},
            },
            "size": top_k,
        }

        response = await self._client.search(index=self._index, body=body)
        return [hit["_source"] for hit in response["hits"]["hits"]]

    def _rrf_merge(
        self, bm25_results: list[dict], knn_results: list[dict], k: int = 60
    ) -> list[dict]:
        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}

        for rank, hit in enumerate(bm25_results):
            cid = hit["chunk_id"]
            scores[cid] = 1 / (k + rank + 1)
            docs[cid] = hit

        for rank, hit in enumerate(knn_results):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
            if cid not in docs:
                docs[cid] = hit

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"score": score, **docs[cid]} for cid, score in ranked]

    def _build_filters(self, filters: dict | None) -> list[dict]:
        conditions: list[dict] = [{"term": {"is_parent": False}}]

        if not filters:
            return conditions

        if filters.get("doc_ids"):
            conditions.append({"terms": {"doc_id": filters["doc_ids"]}})
        if filters.get("language"):
            conditions.append({"term": {"language": filters["language"]}})
        if filters.get("type"):
            conditions.append({"term": {"type": filters["type"]}})
        if filters.get("user_id"):
            conditions.append({"term": {"user_id": filters["user_id"]}})

        return conditions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/vectorstore/test_elasticsearch_store.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/vectorstore/elasticsearch_store.py tests/unit/vectorstore/
git commit -m "feat: implement ElasticsearchStore with hybrid search and RRF"
```

---

## Task 5: VectorStoreFactory

**Files:**
- Create: `backend/app/vectorstore/factory.py`
- Create: `tests/unit/vectorstore/test_factory.py`

- [ ] **Step 1: Write factory test**

Create `tests/unit/vectorstore/test_factory.py`:

```python
import pytest
from unittest.mock import patch

from backend.app.vectorstore.factory import VectorStoreFactory
from backend.app.vectorstore.elasticsearch_store import ElasticsearchStore


@patch("backend.app.vectorstore.factory.ElasticsearchStore")
def test_factory_creates_elasticsearch(mock_es_cls):
    store = VectorStoreFactory.create("elasticsearch")
    mock_es_cls.assert_called_once()


def test_factory_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown vectorstore strategy"):
        VectorStoreFactory.create("pinecone")


@patch("backend.app.vectorstore.factory.ElasticsearchStore")
def test_factory_uses_config_default(mock_es_cls):
    store = VectorStoreFactory.create()
    mock_es_cls.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/vectorstore/test_factory.py -v`
Expected: FAIL — factory module doesn't exist

- [ ] **Step 3: Implement factory**

Create `backend/app/vectorstore/factory.py`:

```python
from backend.app.core.config import settings
from backend.app.pipeline.base.vectorstore import BaseVectorStore


class VectorStoreFactory:

    @staticmethod
    def create(strategy: str | None = None) -> BaseVectorStore:
        strategy = strategy or settings.vectorstore_strategy

        if strategy == "elasticsearch":
            from backend.app.vectorstore.elasticsearch_store import ElasticsearchStore
            return ElasticsearchStore()

        raise ValueError(f"Unknown vectorstore strategy: {strategy}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/vectorstore/test_factory.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/vectorstore/factory.py tests/unit/vectorstore/test_factory.py
git commit -m "feat: add VectorStoreFactory"
```

---

## Task 6: Update Ingestion Service

**Files:**
- Modify: `backend/app/services/ingestion.py`

- [ ] **Step 1: Replace QdrantWrapper with vectorstore in ingestion**

In `backend/app/services/ingestion.py`:

Replace import (line 28):
```python
# Remove:
from backend.app.vectorstore.qdrant_client import QdrantWrapper
# Add:
from backend.app.vectorstore.factory import VectorStoreFactory
```

In `__init__` (line 47), replace:
```python
# Remove:
        self._qdrant = QdrantWrapper()
# Add:
        self._vectorstore = VectorStoreFactory.create()
```

Replace stages 8-9 (lines 152-168) with:

```python
            # 8. Store in Elasticsearch (children with vectors + parents without)
            log.info("stage_store_vectorstore")
            start = time.perf_counter()

            # Set user_id on all chunks
            for chunk in parent_chunks:
                chunk.user_id = user_id or ""
            for chunk in final_children:
                chunk.user_id = user_id or ""

            await self._vectorstore.upsert_chunks(parent_chunks)
            await self._vectorstore.upsert_chunks(final_children, vectors)
            INGESTION_STAGE_DURATION.labels(stage="store_vectorstore").observe(
                time.perf_counter() - start
            )
```

Replace `_store_parents` method (lines 195-234). Remove the `ParentChunk` ORM inserts, keep only the `Document` record:

```python
    async def _store_parents(
        self,
        doc_id: str,
        doc_name: str,
        parent_chunks: list,
        doc_metadata: dict,
        user_id: str = None,
    ):
        async with AsyncSessionLocal() as session:
            doc = Document(
                doc_id=doc_id,
                doc_name=doc_name,
                user_id=user_id,
                language=doc_metadata.get("language", "en"),
                doc_type=doc_metadata.get("doc_type", "document"),
                chunk_count=len(parent_chunks),
                parser_used=doc_metadata.get("parser", ""),
                status="ready",
                metadata_=doc_metadata,
            )
            session.add(doc)
            await session.commit()
```

Note: The `_store_parents` call on line 163-164 stays — it now only stores the Document record. Rename it to `_store_document` and update the call site:

Line 163: `await self._store_document(doc_id, doc_name, parent_chunks, doc_metadata, user_id=user_id)`

Rename the method to `_store_document`.

Update `delete_document` method (lines 236-247):

```python
    async def delete_document(self, doc_id: str):
        await self._vectorstore.delete_by_doc_id(doc_id)
        async with AsyncSessionLocal() as session:
            from sqlalchemy import delete

            await session.execute(
                delete(Document).where(Document.doc_id == doc_id)
            )
            await session.commit()
```

Remove the `ParentChunk` import from line 17:
```python
# Change:
from backend.app.models.document import Document, ParentChunk
# To:
from backend.app.models.document import Document
```

Also update the stage log label from `"stage_store_qdrant"` and `"stage_store_postgres"` to just `"stage_store_vectorstore"`.

- [ ] **Step 2: Run existing tests to check nothing is broken**

Run: `pytest tests/unit/ -v --ignore=tests/unit/vectorstore`
Expected: PASS (if any tests import IngestionService they'll need mocks updated, but unit tests are isolated)

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/ingestion.py
git commit -m "feat: use vectorstore in ingestion service, remove ParentChunk storage"
```

---

## Task 7: Update Retriever Node

**Files:**
- Modify: `backend/app/agent/nodes/retriever.py`
- Modify: `tests/unit/agent/test_retriever.py`

- [ ] **Step 1: Update retriever node**

In `backend/app/agent/nodes/retriever.py`:

Replace imports (lines 11-12, 18, 24):
```python
# Remove:
from sqlalchemy import select
from backend.app.core.database import AsyncSessionLocal
from backend.app.models.document import ParentChunk
from backend.app.vectorstore.qdrant_client import QdrantWrapper

# Add:
from backend.app.vectorstore.factory import VectorStoreFactory
```

Update `_assess_quality` (lines 30-35) — change attribute access to dict access:

```python
def _assess_quality(results: list) -> float:
    """Average score of top-5 results. 0.0 if empty."""
    if not results:
        return 0.0
    scores = [r["score"] for r in results[:5]]
    return sum(scores) / len(scores)
```

Replace `_fetch_parents` function (lines 38-95) — now uses vectorstore:

```python
async def _fetch_parents(child_results: list, vectorstore) -> list[dict]:
    """
    Look up parent chunks from vectorstore.
    For each child, return the parent's full content for richer LLM context.
    Atomic chunks (no parent_id) pass through directly.
    """
    parent_ids = {
        r["parent_id"] for r in child_results if r.get("parent_id")
    }

    parents = {}
    if parent_ids:
        parent_list = await vectorstore.fetch_parents(list(parent_ids))
        parents = {p["chunk_id"]: p for p in parent_list}

    chunks = []
    seen_parents: set[str] = set()

    for r in child_results:
        parent_id = r.get("parent_id")

        if parent_id and parent_id in parents and parent_id not in seen_parents:
            parent = parents[parent_id]
            chunks.append(
                {
                    "content": parent.get("content_raw", ""),
                    "content_markdown": parent.get("content_markdown"),
                    "doc_id": parent.get("doc_id", ""),
                    "doc_name": r.get("doc_name", ""),
                    "page": parent.get("page", 0),
                    "section": parent.get("section", ""),
                    "type": parent.get("type", "text"),
                    "score": r["score"],
                    "chunk_id": parent["chunk_id"],
                }
            )
            seen_parents.add(parent_id)

        elif not parent_id:
            chunks.append(
                {
                    "content": r.get("content_raw", ""),
                    "content_markdown": r.get("content_markdown"),
                    "doc_id": r.get("doc_id", ""),
                    "doc_name": r.get("doc_name", ""),
                    "page": r.get("page", 0),
                    "section": r.get("section", ""),
                    "type": r.get("type", "text"),
                    "score": r["score"],
                    "chunk_id": r.get("chunk_id", ""),
                }
            )

    return chunks
```

Update `retriever_node` function (lines 98-182):

Replace lines 101-102:
```python
    embedder = OpenAIEmbedder()
    vectorstore = VectorStoreFactory.create()
```

Replace the search call (lines 124-128):
```python
        results = await vectorstore.search(
            query_vector=vector,
            query_text=query_text,
            top_k=settings.retrieval_top_k,
            filters=filters if filters else None,
        )
```

Replace the parent fetch call (line 163):
```python
    chunks = await _fetch_parents(results, vectorstore)
```

Update the docstring (lines 1-7) to remove Qdrant/PostgreSQL references.

- [ ] **Step 2: Update retriever tests**

In `tests/unit/agent/test_retriever.py`:

Update `test_assess_quality_*` tests — results are now dicts instead of MagicMock with `.score`:

```python
def test_assess_quality_high_scores():
    results = [{"score": 0.9}, {"score": 0.85}, {"score": 0.8}]
    assert _assess_quality(results) > 0.8


def test_assess_quality_low_scores():
    results = [{"score": 0.3}, {"score": 0.2}]
    assert _assess_quality(results) < 0.4
```

Update `test_retriever_uses_hyde_query` — replace QdrantWrapper mock with vectorstore:

```python
@patch("backend.app.agent.nodes.retriever._fetch_parents")
@patch("backend.app.agent.nodes.retriever.VectorStoreFactory")
@patch("backend.app.agent.nodes.retriever.OpenAIEmbedder")
async def test_retriever_uses_hyde_query(mock_embedder_cls, mock_factory_cls, mock_fetch):
    mock_embedder = AsyncMock()
    mock_embedder_cls.return_value = mock_embedder
    mock_embedder.embed_single = AsyncMock(return_value=[0.1] * 1536)

    mock_store = AsyncMock()
    mock_factory_cls.create.return_value = mock_store
    mock_store.search = AsyncMock(return_value=[
        {"score": 0.9, "parent_id": "p1", "doc_id": "d1", "chunk_id": "c1",
         "content_raw": "text", "doc_name": "test.pdf", "type": "text",
         "page": 1, "section": "s1", "language": "en", "word_count": 50}
    ])

    mock_fetch.return_value = [{"content": "result", "score": 0.9, "chunk_id": "c1"}]

    state = _make_state(hyde_query="hypothetical answer about revenue")
    result = await retriever_node(state)

    assert len(result["retrieved_chunks"]) > 0
    assert result["retrieval_attempts"] >= 1
```

Update `test_retriever_retries_on_low_quality` similarly:

```python
@patch("backend.app.agent.nodes.retriever._fetch_parents")
@patch("backend.app.agent.nodes.retriever.VectorStoreFactory")
@patch("backend.app.agent.nodes.retriever.OpenAIEmbedder")
async def test_retriever_retries_on_low_quality(mock_embedder_cls, mock_factory_cls, mock_fetch):
    mock_embedder = AsyncMock()
    mock_embedder_cls.return_value = mock_embedder
    mock_embedder.embed_single = AsyncMock(return_value=[0.1] * 1536)

    mock_store = AsyncMock()
    mock_factory_cls.create.return_value = mock_store
    mock_store.search = AsyncMock(side_effect=[
        [{"score": 0.3, "parent_id": "p1", "doc_id": "d1", "chunk_id": "c1",
          "content_raw": "text", "doc_name": "test.pdf", "type": "text",
          "page": 1, "section": "s1", "language": "en", "word_count": 50}],
        [{"score": 0.85, "parent_id": "p1", "doc_id": "d1", "chunk_id": "c1",
          "content_raw": "text", "doc_name": "test.pdf", "type": "text",
          "page": 1, "section": "s1", "language": "en", "word_count": 50}],
    ])

    mock_fetch.return_value = [{"content": "result", "score": 0.85, "chunk_id": "c1"}]

    with patch("backend.app.agent.nodes.retriever.get_mini_model") as mock_get_mini:
        mock_llm = MagicMock()
        mock_get_mini.return_value = mock_llm
        mock_bound = AsyncMock()
        mock_llm.bind.return_value = mock_bound
        mock_bound.ainvoke = AsyncMock(
            return_value=MagicMock(content="expanded query")
        )

        state = _make_state()
        result = await retriever_node(state)

        assert result["retrieval_attempts"] == 2
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/unit/agent/test_retriever.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add backend/app/agent/nodes/retriever.py tests/unit/agent/test_retriever.py
git commit -m "feat: use vectorstore in retriever node, dict-based results"
```

---

## Task 8: Update Chunk Viewer API

**Files:**
- Modify: `backend/app/api/chunks.py`

- [ ] **Step 1: Rewrite chunks API to use vectorstore**

Replace the entire file `backend/app/api/chunks.py`:

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select

from backend.app.api.dependencies import get_current_user
from backend.app.core.database import AsyncSessionLocal
from backend.app.models.document import Document
from backend.app.vectorstore.factory import VectorStoreFactory

router = APIRouter()


def _build_chunk_tree(all_chunks: list[dict]) -> list[dict]:
    """Build parent-child tree from flat list of ES documents."""
    parents = [c for c in all_chunks if c.get("is_parent")]
    children = [c for c in all_chunks if not c.get("is_parent")]

    children_by_parent: dict[str, list[dict]] = {}
    for child in children:
        pid = child.get("parent_id", "")
        children_by_parent.setdefault(pid, []).append(child)

    tree = []
    for parent in parents:
        tree.append({
            "chunk_id": parent["chunk_id"],
            "content_raw": parent.get("content_raw", ""),
            "content_markdown": parent.get("content_markdown"),
            "content_html": parent.get("content_html"),
            "type": parent.get("type", "text"),
            "page": parent.get("page", 0),
            "section": parent.get("section", ""),
            "language": parent.get("language", "en"),
            "word_count": parent.get("word_count", 0),
            "children": children_by_parent.get(parent["chunk_id"], []),
        })

    # Add orphan children (atomic chunks with no parent)
    parent_ids = {p["chunk_id"] for p in parents}
    orphan_parent_ids = set(children_by_parent.keys()) - parent_ids
    for pid in orphan_parent_ids:
        if not pid:
            for child in children_by_parent[pid]:
                tree.append({
                    "chunk_id": child["chunk_id"],
                    "content_raw": child.get("content_raw", ""),
                    "content_markdown": child.get("content_markdown"),
                    "content_html": child.get("content_html"),
                    "type": child.get("type", "text"),
                    "page": child.get("page", 0),
                    "section": child.get("section", ""),
                    "language": child.get("language", "en"),
                    "word_count": len(child.get("content_raw", "").split()),
                    "children": [],
                })

    return tree


@router.get("/documents/{doc_id}/chunks")
async def get_document_chunks(
    doc_id: str,
    type_filter: str = Query(default=None, description="Filter by chunk type"),
    page_filter: int = Query(default=None, description="Filter by page number"),
    search: str = Query(default=None, description="Search chunk content"),
    user: dict = Depends(get_current_user),
):
    # Verify document ownership
    async with AsyncSessionLocal() as session:
        doc_result = await session.execute(
            select(Document).where(
                Document.doc_id == doc_id,
                Document.user_id == user["user_id"],
            )
        )
        if not doc_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Document not found")

    # Fetch all chunks from vectorstore
    vectorstore = VectorStoreFactory.create()
    all_chunks = await vectorstore.get_by_doc_id(doc_id)

    # Apply filters
    if type_filter:
        all_chunks = [c for c in all_chunks if c.get("type") == type_filter]
    if page_filter is not None:
        all_chunks = [c for c in all_chunks if c.get("page") == page_filter]
    if search:
        search_lower = search.lower()
        all_chunks = [
            c for c in all_chunks
            if search_lower in c.get("content_raw", "").lower()
            or search_lower in c.get("content", "").lower()
        ]

    tree = _build_chunk_tree(all_chunks)
    return {"doc_id": doc_id, "chunks": tree, "total": len(tree)}
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from backend.app.api.chunks import router; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/app/api/chunks.py
git commit -m "feat: use vectorstore in chunk viewer API"
```

---

## Task 9: Update App Startup

**Files:**
- Modify: `backend/app/main.py`

- [ ] **Step 1: Add vectorstore initialization to lifespan**

In `backend/app/main.py`, add import:

```python
from backend.app.vectorstore.factory import VectorStoreFactory
```

Update the `lifespan` function (lines 14-19):

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_langfuse()
    configure_logging()
    await create_tables()
    vectorstore = VectorStoreFactory.create()
    await vectorstore.initialize()
    yield
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/main.py
git commit -m "feat: initialize vectorstore in app startup"
```

---

## Task 10: Remove Qdrant Code and ParentChunk Model

**Files:**
- Delete: `backend/app/vectorstore/qdrant_client.py`
- Modify: `backend/app/models/document.py`

- [ ] **Step 1: Delete qdrant_client.py**

```bash
rm backend/app/vectorstore/qdrant_client.py
```

- [ ] **Step 2: Remove ParentChunk from models**

In `backend/app/models/document.py`, remove the `ParentChunk` class (lines 29-44). The file should only contain `Base` and `Document`:

```python
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, JSON, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(String, primary_key=True)
    doc_name = Column(String, nullable=False)
    user_id = Column(String, nullable=True, index=True)
    file_path = Column(String)
    language = Column(String, default="en")
    doc_type = Column(String, default="document")
    page_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    parser_used = Column(String)
    status = Column(String, default="processing")
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

(Note: removed unused `Boolean` and `Text` imports too since `ParentChunk` was the only user.)

- [ ] **Step 3: Run all unit tests to verify nothing is broken**

Run: `pytest tests/unit/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git rm backend/app/vectorstore/qdrant_client.py
git add backend/app/models/document.py
git commit -m "refactor: remove QdrantWrapper and ParentChunk model"
```

---

## Task 11: Update Integration Tests

**Files:**
- Modify: `tests/integration/test_ingestion_pipeline.py`
- Modify: `tests/integration/test_chat_pipeline.py`

- [ ] **Step 1: Update ingestion integration test**

Replace `tests/integration/test_ingestion_pipeline.py`:

```python
"""
Requires: Elasticsearch + PostgreSQL running (docker compose up elasticsearch postgres)
Run with: pytest tests/integration/ -m integration
"""
import pytest


@pytest.mark.integration
async def test_full_ingestion_pipeline(sample_pdf_path):
    from backend.app.services.ingestion import IngestionService

    service = IngestionService()
    result = await service.ingest(
        file_path=sample_pdf_path,
        doc_name="sample.pdf",
        language="en",
    )

    assert result["doc_id"] is not None
    assert result["child_chunks"] > 0
    assert result["parent_chunks"] > 0
    assert result["child_chunks"] >= result["parent_chunks"]


@pytest.mark.integration
async def test_delete_removes_from_stores(sample_pdf_path):
    from backend.app.services.ingestion import IngestionService
    from backend.app.vectorstore.factory import VectorStoreFactory

    service = IngestionService()
    result = await service.ingest(sample_pdf_path, "test.pdf")
    doc_id = result["doc_id"]

    await service.delete_document(doc_id)

    vectorstore = VectorStoreFactory.create()
    remaining = await vectorstore.get_by_doc_id(doc_id)
    assert len(remaining) == 0
```

- [ ] **Step 2: Update chat integration test docstring**

In `tests/integration/test_chat_pipeline.py`, change line 2:

```python
# From:
"""
Requires: Qdrant + PostgreSQL running (docker compose up qdrant postgres)
# To:
"""
Requires: Elasticsearch + PostgreSQL running (docker compose up elasticsearch postgres)
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_ingestion_pipeline.py tests/integration/test_chat_pipeline.py
git commit -m "test: update integration tests for Elasticsearch"
```

---

## Task 12: Update Documentation and Scripts

**Files:**
- Modify: `scripts/seed_demo_data.py`
- Modify: `scripts/seed_custom_eval.py`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update seed script docstrings**

In `scripts/seed_demo_data.py`, line 6:
```python
# From:
Requires: Docker services running (qdrant, postgres, redis)
# To:
Requires: Docker services running (elasticsearch, postgres, redis)
```

In `scripts/seed_custom_eval.py`, line 6:
```python
# From:
Requires: Docker services running (qdrant, postgres, redis)
# To:
Requires: Docker services running (elasticsearch, postgres, redis)
```

- [ ] **Step 2: Update CLAUDE.md**

In `CLAUDE.md`, update the Key Directories section:
- Change `backend/app/vectorstore/qdrant_client.py` reference to `backend/app/vectorstore/elasticsearch_store.py`
- Change description from `QdrantWrapper (upsert, search, delete, filtering)` to `ElasticsearchStore (hybrid search, RRF, upsert, delete)`

In the Infrastructure section:
- Change `docker compose up -d` comment to mention `elasticsearch` instead of `qdrant`
- Update Docker Services to replace `Qdrant on 6333` with `Elasticsearch on 9200`

In the Architecture section, update Pipeline Flow:
```
→ Qdrant (child vectors) + PostgreSQL (parent chunks + metadata)
```
to:
```
→ Elasticsearch (child vectors + BM25 + parent chunks) + PostgreSQL (document metadata)
```

- [ ] **Step 3: Commit**

```bash
git add scripts/seed_demo_data.py scripts/seed_custom_eval.py CLAUDE.md
git commit -m "docs: update references from Qdrant to Elasticsearch"
```

---

## Task 13: Final Verification

- [ ] **Step 1: Run full unit test suite**

Run: `pytest tests/unit/ -v`
Expected: ALL PASS

- [ ] **Step 2: Run linting**

Run: `ruff check . && ruff format --check .`
Expected: No errors (fix any issues found)

- [ ] **Step 3: Run type checking**

Run: `mypy backend/ --ignore-missing-imports`
Expected: No new errors

- [ ] **Step 4: Verify docker compose is valid**

Run: `docker compose config --quiet`
Expected: No errors

- [ ] **Step 5: Final commit if any lint fixes needed**

```bash
git add -A
git commit -m "fix: lint and formatting fixes"
```
