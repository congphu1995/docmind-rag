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
