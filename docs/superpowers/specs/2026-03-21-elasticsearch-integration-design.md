# Elasticsearch Integration — Design Spec

> Replace Qdrant + PostgreSQL parent chunk storage with Elasticsearch.
> Hybrid search (dense vectors + BM25) via application-side RRF.

| Field | Value |
|---|---|
| Date | 2026-03-21 |
| Scope | Option B — ES replaces Qdrant (child vectors) + PostgreSQL (parent chunks). PostgreSQL retains Document + User models. |
| Hybrid strategy | RRF (Reciprocal Rank Fusion), application-side |
| BM25 target | `content_raw` (original text) |
| Dense target | `embedding` (from enriched `content`) |
| Retry logic | Unchanged — AdaptiveRetriever evolves based on eval data |
| Architecture | BaseVectorStore ABC + Factory (Approach 2) |

---

## 0. Chunk Dataclass Changes

Before implementing the vector store, the `Chunk` dataclass (`backend/app/pipeline/base/chunker.py`) needs two changes:

### 0.1 Add `user_id` field

Currently `user_id` is only set on the PostgreSQL `ParentChunk` ORM object during ingestion, bypassing the `Chunk` dataclass. With ES replacing PostgreSQL for parent storage, `user_id` must live on the `Chunk` itself so `upsert_chunks()` can include it.

```python
@dataclass
class Chunk:
    # ... existing fields ...
    user_id: str = ""          # NEW — set during ingestion, required for per-user filtering
```

The ingestion service sets `user_id` on all chunks (parents and children) before calling `vectorstore.upsert_chunks()`.

### 0.2 Rename `qdrant_payload()` to `to_document()`

The current `qdrant_payload()` method is Qdrant-specific and omits fields needed by ES (`content_html`, `is_parent`, `user_id`, `created_at`). Rename to a generic serialization method:

```python
def to_document(self) -> dict:
    """Serialize chunk to a dict suitable for any vector store."""
    doc = {
        "chunk_id": self.chunk_id,
        "parent_id": self.parent_id,
        "doc_id": self.doc_id,
        "doc_name": self.doc_name,
        "user_id": self.user_id,
        "content": self.content,               # enriched text (used for embedding)
        "content_raw": self.content_raw,
        "content_markdown": self.content_markdown,
        "content_html": self.content_html,
        "type": self.type,
        "page": self.page,
        "section": self.section,
        "language": self.language,
        "word_count": self.word_count,
        "is_parent": self.is_parent,
        "metadata": self.metadata,             # nested object, not flattened
        "created_at": datetime.utcnow().isoformat(),
    }
    return doc
```

Note: `content_html` was previously only stored on PostgreSQL parents. Now both parents and children include it in ES (new capability, not just migration).

---

## 1. ES Index Schema

Single index `docmind_chunks` stores both parents and children.

```json
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
    "properties": {
      "chunk_id":         { "type": "keyword" },
      "parent_id":        { "type": "keyword" },
      "doc_id":           { "type": "keyword" },
      "doc_name":         { "type": "keyword" },
      "user_id":          { "type": "keyword" },

      "content":          { "type": "text", "index": false },
      "content_raw":      { "type": "text", "analyzer": "standard" },
      "content_markdown": { "type": "text", "index": false },
      "content_html":     { "type": "text", "index": false },

      "embedding":        { "type": "dense_vector", "dims": 1536,
                            "index": true, "similarity": "cosine" },

      "type":             { "type": "keyword" },
      "page":             { "type": "integer" },
      "section":          { "type": "keyword" },
      "language":         { "type": "keyword" },
      "word_count":       { "type": "integer" },
      "is_parent":        { "type": "boolean" },
      "metadata":         { "type": "object", "dynamic": true },
      "created_at":       { "type": "date" }
    }
  }
}
```

- **`content_raw`** — BM25-indexed with `standard` analyzer. Keyword search target.
- **`embedding`** — Dense vector for children only. Parents have no embedding (field absent), so they don't participate in vector search.
- **`content`** — Enriched text (post-`ContextEnricher`). Stored but not BM25-indexed. Kept for audit: shows what text was embedded.
- **`content_markdown` / `content_html`** — Stored but not BM25-indexed (`index: false`). Served to LLM and UI.
- **`is_parent`** — Boolean flag distinguishing parents from children in the same index.
- **`metadata`** — Nested object (not flattened to top level) for extensible fields (doc_type, date, org). Avoids key collisions with reserved fields like `type` or `page`.

---

## 2. BaseVectorStore ABC

New file: `backend/app/pipeline/base/vectorstore.py`

```python
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
        filters: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """Search for similar chunks. Returns scored child chunks."""
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

`search()` takes both `query_vector` and `query_text`. ES uses both for hybrid search. A future Qdrant implementation would ignore `query_text`. Hybrid search is an implementation detail, not a leaking abstraction.

`upsert_chunks()` accepts optional vectors. Parents stored without vectors, children with.

---

## 3. ElasticsearchStore Implementation

File: `backend/app/vectorstore/elasticsearch_store.py`

### 3.1 Hybrid Search with Application-Side RRF

Two separate queries merged in Python. Guaranteed compatible with ES free Basic license.

```python
async def search(self, query_vector, query_text, top_k, filters, score_threshold):
    # Fetch larger candidate set for better RRF fusion
    candidate_k = top_k * 3

    # 1. BM25 query on content_raw (children only)
    bm25_results = await self._bm25_search(query_text, candidate_k, filters)

    # 2. kNN query on embedding (children only)
    knn_results = await self._knn_search(query_vector, candidate_k, filters)

    # 3. Merge with RRF
    merged = self._rrf_merge(bm25_results, knn_results, k=60)

    # 4. Return top_k (score_threshold not applied — see note below)
    return merged[:top_k]
```

Both queries filter to `is_parent: false` plus user-provided filters (doc_ids, language, type, user_id).

**Note on `score_threshold`:** The `score_threshold` parameter is accepted for interface compatibility but **not applied** by `ElasticsearchStore`. RRF scores are on a completely different scale than cosine similarity (typically 0.0-0.03 vs 0.0-1.0). The existing threshold of 0.4 is meaningless for RRF. Quality assessment in the `AdaptiveRetriever` continues to work because it uses the top-5 average of whatever scores are returned, which adapts naturally to any scale.

### 3.2 RRF Merge

```python
def _rrf_merge(self, bm25_results, knn_results, k=60):
    scores = {}
    docs = {}
    for rank, hit in enumerate(bm25_results):
        cid = hit["chunk_id"]
        scores[cid] = 1 / (k + rank + 1)
        docs[cid] = hit
    for rank, hit in enumerate(knn_results):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        if cid not in docs:
            docs[cid] = hit
    # Sort by combined RRF score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"score": score, **docs[cid]} for cid, score in ranked]
```

### 3.3 Return Schema from `search()`

All results from `search()` are dicts with this shape (used by `AdaptiveRetriever` for quality assessment and parent expansion):

```python
{
    "score": float,          # RRF score (NOT cosine — different scale)
    "chunk_id": str,
    "parent_id": str | None,
    "doc_id": str,
    "doc_name": str,
    "content_raw": str,
    "type": str,
    "page": int,
    "section": str,
    "language": str,
    "word_count": int,
    **metadata
}
```

The retriever's `_assess_quality()` currently accesses `r.score` (attribute on Qdrant `ScoredPoint`). This must change to `r["score"]` (dict access). Similarly, `_fetch_parents()` accesses `r.payload.get("parent_id")` which becomes `r["parent_id"]`.

### 3.4 Other Methods

- **`upsert_chunks()`** — ES `_bulk` API. Batch size: 100 (configurable via `es_bulk_batch_size` setting). Children include `embedding` field, parents don't. Serializes chunks via `chunk.to_document()`.
- **`fetch_parents()`** — `terms` query on `chunk_id` where `is_parent: true`.
- **`delete_by_doc_id()`** — `delete_by_query` with `doc_id` filter.
- **`get_by_doc_id()`** — `search` with `doc_id` filter, no scoring, returns all parents + children.
- **`initialize()`** — Create index with mapping + settings from Section 1 if it doesn't exist.

### 3.5 Error Handling

All ES operations wrap exceptions in the existing `VectorStoreError` (from `backend/app/core/exceptions.py`). Common cases:
- Connection refused → `VectorStoreError("Elasticsearch unavailable")`
- Index not found → auto-create via `initialize()`, retry
- Bulk partial failure → log failed items, raise `VectorStoreError` with count

### 3.6 Client

Uses `elasticsearch[async]` — the official async Python client (`AsyncElasticsearch`).

Config includes optional auth for production:
```python
elasticsearch_username: str = ""    # empty = no auth (dev)
elasticsearch_password: str = ""
```

---

## 4. Integration Points

### 4.1 Config (`backend/app/core/config.py`)

```python
vectorstore_strategy: str = "elasticsearch"
elasticsearch_url: str = "http://localhost:9200"
elasticsearch_index: str = "docmind_chunks"
elasticsearch_username: str = ""
elasticsearch_password: str = ""
rrf_k: int = 60
es_bulk_batch_size: int = 100
```

Existing Qdrant settings kept (not breaking), just unused when strategy is `elasticsearch`.

### 4.2 Factory (`backend/app/vectorstore/factory.py`)

```python
class VectorStoreFactory:
    @staticmethod
    def create(strategy: str = None) -> BaseVectorStore:
        strategy = strategy or settings.vectorstore_strategy
        if strategy == "elasticsearch":
            return ElasticsearchStore(...)
        raise ValueError(f"Unknown vectorstore strategy: {strategy}")
```

### 4.3 Ingestion Service (`backend/app/services/ingestion.py`)

Current stages 8-9 (Qdrant upsert + PostgreSQL parent insert) collapse:
- Set `user_id` on all chunks before storing
- `vectorstore.upsert_chunks(parents)` — no vectors
- `vectorstore.upsert_chunks(children, vectors)` — with vectors
- Remove direct `ParentChunk` ORM inserts

**`delete_document()`** also needs updating: currently deletes from both Qdrant and PostgreSQL (`ParentChunk`). After migration, call `vectorstore.delete_by_doc_id()` (deletes both parents and children from ES) plus the existing PostgreSQL `Document` record deletion.

### 4.4 Retriever Node (`backend/app/agent/nodes/retriever.py`)

- `QdrantWrapper.search(vector, ...)` becomes `vectorstore.search(vector, query_text, ...)`
- PostgreSQL parent query becomes `vectorstore.fetch_parents(parent_ids)`
- `query_text` comes from agent state `rewritten_query`
- `_assess_quality()` changes from `r.score` (attribute) to `r["score"]` (dict access)
- `_fetch_parents()` changes from `r.payload.get("parent_id")` to `r["parent_id"]`

### 4.5 Chunk Viewer API (`backend/app/api/chunks.py`)

- Separate Qdrant + PostgreSQL queries become single `vectorstore.get_by_doc_id()`
- `_build_chunk_tree()` currently accesses parents via ORM attributes (`parent.chunk_id`, `parent.content_raw`). With ES, parents are dicts — change to dict access (`parent["chunk_id"]`, `parent["content_raw"]`), consistent with how children are already handled.
- Same response shape, simpler implementation

### 4.6 App Startup (`backend/app/main.py`)

- Call `vectorstore.initialize()` in FastAPI lifespan to ensure index exists

---

## 5. Infrastructure & Migration

### 5.1 Docker Compose — Add Elasticsearch

```yaml
elasticsearch:
  image: elasticsearch:8.15.0
  environment:
    - discovery.type=single-node
    - xpack.security.enabled=false
    - xpack.security.http.ssl.enabled=false
    - ES_JAVA_OPTS=-Xms512m -Xmx512m
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

Single node, security disabled for dev. Free Basic license includes kNN + full-text search.

Backend/worker use `depends_on: elasticsearch: condition: service_healthy` to wait for ES startup (30-60s).

### 5.2 Remove Qdrant

- Remove `qdrant` service and `qdrant_data` volume from docker-compose
- Backend/worker `depends_on` switch from `qdrant` to `elasticsearch`

### 5.3 Python Dependencies

- Add: `elasticsearch[async]`
- Remove: `qdrant-client`

### 5.4 Code Removed

| File | Action |
|---|---|
| `backend/app/vectorstore/qdrant_client.py` | Delete — replaced by `elasticsearch_store.py` |
| `backend/app/models/document.py` — `ParentChunk` class | Remove — parents now in ES |
| `backend/app/core/database.py` — `ParentChunk` table creation | Remove |

### 5.5 Unchanged

- `Document` and `User` models in PostgreSQL
- `AdaptiveRetriever` retry logic
- All chunking, parsing, embedding pipeline
- `BaseRetriever` ABC (separate concern)
- Frontend (same API contracts)

### 5.6 Migration Path

No data migration tool. Re-ingest documents after switching. The ingestion pipeline derives chunks from raw files — chunks are not the source of truth.

---

## File Summary

### New Files

| File | Action |
|---|---|
| `backend/app/pipeline/base/vectorstore.py` | **New** — BaseVectorStore ABC |
| `backend/app/vectorstore/elasticsearch_store.py` | **New** — ElasticsearchStore implementation |
| `backend/app/vectorstore/factory.py` | **New** — VectorStoreFactory |

### Core Edits

| File | Action |
|---|---|
| `backend/app/pipeline/base/chunker.py` | **Edit** — add `user_id` field, rename `qdrant_payload()` → `to_document()` |
| `backend/app/core/config.py` | **Edit** — add ES settings |
| `backend/app/services/ingestion.py` | **Edit** — use vectorstore for parents + children, update `delete_document()` |
| `backend/app/agent/nodes/retriever.py` | **Edit** — use vectorstore.search() + fetch_parents(), fix dict access |
| `backend/app/api/chunks.py` | **Edit** — use vectorstore.get_by_doc_id() |
| `backend/app/main.py` | **Edit** — initialize vectorstore in lifespan |
| `backend/app/models/document.py` | **Edit** — remove ParentChunk |
| `backend/app/core/database.py` | **Edit** — remove ParentChunk table creation |

### Infrastructure

| File | Action |
|---|---|
| `docker-compose.yml` | **Edit** — add ES with healthcheck, remove Qdrant |
| `pyproject.toml` | **Edit** — add elasticsearch[async], remove qdrant-client |
| `.env.example` | **Edit** — add `ELASTICSEARCH_URL`, remove `QDRANT_HOST`/`QDRANT_PORT` |

### Tests

| File | Action |
|---|---|
| `tests/unit/pipeline/test_schemas.py` | **Edit** — update `qdrant_payload()` tests → `to_document()` |
| `tests/unit/agent/test_retriever.py` | **Edit** — update QdrantWrapper patches to use vectorstore |
| `tests/integration/test_ingestion_pipeline.py` | **Edit** — update QdrantWrapper imports/references |
| `tests/integration/test_chat_pipeline.py` | **Edit** — update Qdrant references |

### Delete

| File | Action |
|---|---|
| `backend/app/vectorstore/qdrant_client.py` | **Delete** — replaced by elasticsearch_store.py |

### Documentation (update references)

| File | Action |
|---|---|
| `CLAUDE.md` | **Edit** — update vectorstore references |
| `scripts/seed_demo_data.py` | **Edit** — update "Requires: qdrant" docstring |
| `scripts/seed_custom_eval.py` | **Edit** — update "Requires: qdrant" docstring |
