import uuid

from qdrant_client import QdrantClient as _QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from backend.app.core.config import settings
from backend.app.core.exceptions import VectorStoreError
from backend.app.core.logging import logger


class QdrantWrapper:

    def __init__(self):
        self._client = _QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self._collection = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dimensions,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("qdrant_collection_created", collection=self._collection)

    async def upsert(self, chunks: list, vectors: list[list[float]]):
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=chunk.qdrant_payload(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        batch_size = 100
        for i in range(0, len(points), batch_size):
            try:
                self._client.upsert(
                    collection_name=self._collection,
                    points=points[i : i + batch_size],
                )
            except Exception as e:
                raise VectorStoreError(f"Qdrant upsert failed: {e}") from e

        logger.info("qdrant_upsert_done", count=len(points))

    async def search(
        self,
        vector: list[float],
        top_k: int = 20,
        score_threshold: float = None,
        filters: dict = None,
    ) -> list[ScoredPoint]:
        qdrant_filter = self._build_filter(filters) if filters else None

        try:
            results = self._client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=top_k,
                score_threshold=score_threshold or settings.retrieval_score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        except Exception as e:
            raise VectorStoreError(f"Qdrant search failed: {e}") from e

        return results

    async def delete_by_doc_id(self, doc_id: str):
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                ]
            ),
        )
        logger.info("qdrant_delete_done", doc_id=doc_id)

    def _build_filter(self, filters: dict) -> Filter:
        conditions = []

        if "doc_ids" in filters and filters["doc_ids"]:
            conditions.append(
                FieldCondition(
                    key="doc_id", match=MatchAny(any=filters["doc_ids"])
                )
            )
        if "language" in filters:
            conditions.append(
                FieldCondition(
                    key="language", match=MatchValue(value=filters["language"])
                )
            )
        if "type" in filters:
            conditions.append(
                FieldCondition(
                    key="type", match=MatchValue(value=filters["type"])
                )
            )

        return Filter(must=conditions) if conditions else None
