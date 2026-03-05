from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import uuid

from backend.app.pipeline.base.parser import ParsedElement


@dataclass
class Chunk:
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    doc_id: str = ""
    doc_name: str = ""

    content: str = ""
    content_raw: str = ""
    content_markdown: Optional[str] = None
    content_html: Optional[str] = None

    type: str = "text"
    page: int = 0
    section: str = ""
    language: str = "en"
    word_count: int = 0
    is_parent: bool = False

    metadata: dict = field(default_factory=dict)

    def qdrant_payload(self) -> dict:
        """Flat dict for Qdrant payload storage."""
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "content_raw": self.content_raw,
            "content_markdown": self.content_markdown,
            "type": self.type,
            "page": self.page,
            "section": self.section,
            "language": self.language,
            "word_count": self.word_count,
            **self.metadata,
        }


class BaseChunker(ABC):
    @abstractmethod
    def chunk(
        self,
        elements: list[ParsedElement],
        doc_metadata: dict,
    ) -> tuple[list[Chunk], list[Chunk]]:
        """
        Returns (parent_chunks, child_chunks).
        parent_chunks → PostgreSQL
        child_chunks  → Qdrant (embedded)
        """
        ...
