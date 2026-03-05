import uuid

from backend.app.core.config import settings
from backend.app.pipeline.base.chunker import BaseChunker, Chunk
from backend.app.pipeline.base.parser import ElementType, ParsedElement


class ParentChildChunker(BaseChunker):
    """
    Splits text elements into parent-child hierarchy.
    Parent (~800 words) → PostgreSQL, sent to LLM
    Child  (~150 words) → Qdrant, used for retrieval

    Tables, figures, code → always atomic (single chunk, no splitting)
    Titles → section boundary marker, not a chunk
    """

    def __init__(
        self,
        parent_size: int = None,
        child_size: int = None,
        overlap: int = None,
    ):
        self.parent_size = parent_size or settings.parent_chunk_size
        self.child_size = child_size or settings.child_chunk_size
        self.overlap = overlap or settings.chunk_overlap

    def chunk(
        self,
        elements: list[ParsedElement],
        doc_metadata: dict,
    ) -> tuple[list[Chunk], list[Chunk]]:
        parent_chunks: list[Chunk] = []
        child_chunks: list[Chunk] = []

        current_section = "Introduction"
        text_buffer: list[ParsedElement] = []

        for el in elements:
            if el.is_structural_boundary():
                if text_buffer:
                    parents, children = self._flush_text_buffer(
                        text_buffer, current_section, doc_metadata
                    )
                    parent_chunks.extend(parents)
                    child_chunks.extend(children)
                    text_buffer = []
                current_section = el.content.strip("# ").strip()

            elif el.is_atomic():
                if text_buffer:
                    parents, children = self._flush_text_buffer(
                        text_buffer, current_section, doc_metadata
                    )
                    parent_chunks.extend(parents)
                    child_chunks.extend(children)
                    text_buffer = []
                atomic = self._make_atomic_chunk(el, current_section, doc_metadata)
                parent_chunks.append(atomic)
                child_chunks.append(atomic)

            else:
                text_buffer.append(el)
                current_words = sum(
                    self._count_words(e.content) for e in text_buffer
                )
                if current_words >= self.parent_size * 1.2:
                    parents, children = self._flush_text_buffer(
                        text_buffer, current_section, doc_metadata
                    )
                    parent_chunks.extend(parents)
                    child_chunks.extend(children)
                    text_buffer = []

        if text_buffer:
            parents, children = self._flush_text_buffer(
                text_buffer, current_section, doc_metadata
            )
            parent_chunks.extend(parents)
            child_chunks.extend(children)

        return parent_chunks, child_chunks

    def _flush_text_buffer(
        self,
        buffer: list[ParsedElement],
        section: str,
        doc_metadata: dict,
    ) -> tuple[list[Chunk], list[Chunk]]:
        if not buffer:
            return [], []

        full_text = "\n\n".join(el.content for el in buffer)
        page = buffer[0].page
        doc_id = buffer[0].doc_id
        doc_name = buffer[0].doc_name
        language = buffer[0].language or "en"

        parent_id = str(uuid.uuid4())
        parent = Chunk(
            chunk_id=parent_id,
            parent_id=None,
            doc_id=doc_id,
            doc_name=doc_name,
            content=full_text,
            content_raw=full_text,
            type="text",
            page=page,
            section=section,
            language=language,
            word_count=self._count_words(full_text),
            is_parent=True,
            metadata=doc_metadata,
        )

        children = []
        child_texts = self._split_into_children(full_text)

        for i, child_text in enumerate(child_texts):
            child = Chunk(
                chunk_id=str(uuid.uuid4()),
                parent_id=parent_id,
                doc_id=doc_id,
                doc_name=doc_name,
                content=child_text,
                content_raw=child_text,
                type="text",
                page=page,
                section=section,
                language=language,
                word_count=self._count_words(child_text),
                is_parent=False,
                metadata={**doc_metadata, "chunk_index": i},
            )
            children.append(child)

        return [parent], children

    def _make_atomic_chunk(
        self,
        el: ParsedElement,
        section: str,
        doc_metadata: dict,
    ) -> Chunk:
        content = el.content
        if el.type == ElementType.TABLE and el.table_html:
            content_markdown = el.content
            content_html = el.table_html
        else:
            content_markdown = None
            content_html = None

        return Chunk(
            chunk_id=str(uuid.uuid4()),
            parent_id=None,
            doc_id=el.doc_id,
            doc_name=el.doc_name,
            content=content,
            content_raw=content,
            content_markdown=content_markdown,
            content_html=content_html,
            type=el.type.value,
            page=el.page,
            section=section,
            language=el.language or "en",
            word_count=self._count_words(content),
            is_parent=True,
            metadata=doc_metadata,
        )

    def _split_into_children(self, text: str) -> list[str]:
        """Split text into child-sized chunks by word count with overlap."""
        words = text.split()
        if len(words) <= self.child_size:
            return [text]

        children = []
        start = 0
        while start < len(words):
            end = min(start + self.child_size, len(words))
            child_text = " ".join(words[start:end])
            children.append(child_text)
            start += self.child_size - self.overlap

        return children

    @staticmethod
    def _count_words(text: str) -> int:
        return len(text.split())
