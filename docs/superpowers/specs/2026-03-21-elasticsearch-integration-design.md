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

## 1. ES Index Schema

Single index `docmind_chunks` stores both parents and children.

```json
{
  "mappings": {
    "properties": {
      "chunk_id":         { "type": "keyword" },
      "parent_id":        { "type": "keyword" },
      "doc_id":           { "type": "keyword" },
      "doc_name":         { "type": "keyword" },
      "user_id":          { "type": "keyword" },

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
- **`content_markdown` / `content_html`** — Stored but not BM25-indexed (`index: false`). Served to LLM and UI.
- **`is_parent`** — Boolean flag distinguishing parents from children in the same index.
- **`metadata`** — Dynamic object for extensible fields (doc_type, date, org, etc.).

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
        """Fetch parent chunks by their IDs."""
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
    # 1. BM25 query on content_raw (children only)
    bm25_results = await self._bm25_search(query_text, top_k, filters)

    # 2. kNN query on embedding (children only)
    knn_results = await self._knn_search(query_vector, top_k, filters)

    # 3. Merge with RRF
    merged = self._rrf_merge(bm25_results, knn_results, k=60)

    # 4. Apply score threshold, return top_k
    return merged[:top_k]
```

Both queries filter to `is_parent: false` plus user-provided filters (doc_ids, language, type, user_id).

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

### 3.3 Other Methods

- **`upsert_chunks()`** — ES `_bulk` API. Children include `embedding` field, parents don't.
- **`fetch_parents()`** — `terms` query on `chunk_id` where `is_parent: true`.
- **`delete_by_doc_id()`** — `delete_by_query` with `doc_id` filter.
- **`get_by_doc_id()`** — `search` with `doc_id` filter, no scoring, returns all parents + children.
- **`initialize()`** — Create index with mapping from Section 1 if it doesn't exist.

### 3.4 Client

Uses `elasticsearch[async]` — the official async Python client (`AsyncElasticsearch`).

---

## 4. Integration Points

### 4.1 Config (`backend/app/core/config.py`)

```python
vectorstore_strategy: str = "elasticsearch"
elasticsearch_url: str = "http://localhost:9200"
elasticsearch_index: str = "docmind_chunks"
rrf_k: int = 60
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
- `vectorstore.upsert_chunks(parents)` — no vectors
- `vectorstore.upsert_chunks(children, vectors)` — with vectors
- Remove direct `ParentChunk` ORM inserts

### 4.4 Retriever Node (`backend/app/agent/nodes/retriever.py`)

- `QdrantWrapper.search(vector, ...)` becomes `vectorstore.search(vector, query_text, ...)`
- PostgreSQL parent query becomes `vectorstore.fetch_parents(parent_ids)`
- `query_text` comes from agent state `rewritten_query`

### 4.5 Chunk Viewer API (`backend/app/api/chunks.py`)

- Separate Qdrant + PostgreSQL queries become single `vectorstore.get_by_doc_id()`
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
```

Single node, security disabled for dev. Free Basic license includes kNN + full-text search.

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

| File | Action |
|---|---|
| `backend/app/pipeline/base/vectorstore.py` | **New** — BaseVectorStore ABC |
| `backend/app/vectorstore/elasticsearch_store.py` | **New** — ElasticsearchStore implementation |
| `backend/app/vectorstore/factory.py` | **New** — VectorStoreFactory |
| `backend/app/core/config.py` | **Edit** — add ES settings |
| `backend/app/services/ingestion.py` | **Edit** — use vectorstore for parents + children |
| `backend/app/agent/nodes/retriever.py` | **Edit** — use vectorstore.search() + fetch_parents() |
| `backend/app/api/chunks.py` | **Edit** — use vectorstore.get_by_doc_id() |
| `backend/app/main.py` | **Edit** — initialize vectorstore in lifespan |
| `docker-compose.yml` | **Edit** — add ES, remove Qdrant |
| `pyproject.toml` | **Edit** — add elasticsearch[async], remove qdrant-client |
| `backend/app/vectorstore/qdrant_client.py` | **Delete** |
| `backend/app/models/document.py` | **Edit** — remove ParentChunk |
