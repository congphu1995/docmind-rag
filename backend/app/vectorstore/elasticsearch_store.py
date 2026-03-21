"""
Elasticsearch vector store — hybrid search (dense + BM25) with RRF fusion.
"""

from elasticsearch import AsyncElasticsearch

from backend.app.core.config import settings
from backend.app.core.exceptions import VectorStoreError
from backend.app.core.logging import logger
from backend.app.pipeline.base.chunker import Chunk
from backend.app.pipeline.base.vectorstore import BaseVectorStore

INDEX_SETTINGS = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
}

INDEX_MAPPINGS = {
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
                    index=self._index,
                    settings=INDEX_SETTINGS,
                    mappings=INDEX_MAPPINGS,
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
                query={
                    "bool": {
                        "filter": [
                            {"terms": {"chunk_id": parent_ids}},
                            {"term": {"is_parent": True}},
                        ]
                    }
                },
                size=len(parent_ids),
            )
        except Exception as e:
            raise VectorStoreError(f"ES fetch_parents failed: {e}") from e

        return [hit["_source"] for hit in response["hits"]["hits"]]

    async def delete_by_doc_id(self, doc_id: str) -> None:
        try:
            await self._client.delete_by_query(
                index=self._index,
                query={"term": {"doc_id": doc_id}},
            )
            logger.info("es_delete_done", doc_id=doc_id)
        except Exception as e:
            raise VectorStoreError(f"ES delete failed: {e}") from e

    async def get_by_doc_id(self, doc_id: str) -> list[dict]:
        try:
            response = await self._client.search(
                index=self._index,
                query={"term": {"doc_id": doc_id}},
                size=10000,
            )
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.warning("es_get_by_doc_failed", error=str(e))
            return []

    # ── Private helpers ──────────────────────────────────────────

    async def _bm25_search(
        self, query_text: str, top_k: int, filters: dict | None
    ) -> list[dict]:
        response = await self._client.search(
            index=self._index,
            query={
                "bool": {
                    "must": [{"match": {"content_raw": query_text}}],
                    "filter": self._build_filters(filters),
                }
            },
            size=top_k,
        )
        return [hit["_source"] for hit in response["hits"]["hits"]]

    async def _knn_search(
        self, query_vector: list[float], top_k: int, filters: dict | None
    ) -> list[dict]:
        response = await self._client.search(
            index=self._index,
            knn={
                "field": "embedding",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": top_k * 5,
                "filter": {"bool": {"filter": self._build_filters(filters)}},
            },
            size=top_k,
        )
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
