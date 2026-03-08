import uuid

from backend.app.core.config import settings
from backend.app.pipeline.base.chunker import BaseChunker, Chunk
from backend.app.pipeline.base.parser import ElementType, ParsedElement
from backend.app.pipeline.chunkers.sentence_splitter import split_sentences


class ParentChildChunker(BaseChunker):
    """
    Semantic chunking with parent-child hierarchy.

    Parents: section-based (one heading = one parent), with merge/split guardrails.
    Children: paragraph-based within each parent, with merge/split guardrails.
    Atomic elements (tables, figures, code): never split, single parent+child.
    Titles: section boundary markers, not chunks themselves.
    """

    def __init__(
        self,
        parent_min_words: int | None = None,
        parent_max_words: int | None = None,
        child_min_words: int | None = None,
        child_max_words: int | None = None,
    ):
        self.parent_min_words = (
            parent_min_words
            if parent_min_words is not None
            else settings.parent_min_words
        )
        self.parent_max_words = (
            parent_max_words
            if parent_max_words is not None
            else settings.parent_max_words
        )
        self.child_min_words = (
            child_min_words if child_min_words is not None else settings.child_min_words
        )
        self.child_max_words = (
            child_max_words if child_max_words is not None else settings.child_max_words
        )

    # ── public API ──────────────────────────────────────────────

    def chunk(
        self,
        elements: list[ParsedElement],
        doc_metadata: dict,
    ) -> tuple[list[Chunk], list[Chunk]]:
        sections = self._collect_sections(elements)
        sections = self._merge_small_sections(sections)

        all_parents: list[Chunk] = []
        all_children: list[Chunk] = []

        for section in sections:
            if section["atomic"]:
                atomic = self._make_atomic_chunk(
                    section["atomic"], section["title"], doc_metadata
                )
                all_parents.append(atomic)
                all_children.append(atomic)
            else:
                parents, children = self._build_parent_children(section, doc_metadata)
                all_parents.extend(parents)
                all_children.extend(children)

        return all_parents, all_children

    # ── section collection ──────────────────────────────────────

    def _collect_sections(self, elements: list[ParsedElement]) -> list[dict]:
        """
        Walk elements, group by title boundaries.
        Returns list of section dicts:
          {"title": str, "elements": [ParsedElement], "atomic": ParsedElement|None}
        Atomic elements (table/figure/code) become their own section.
        """
        sections: list[dict] = []
        current_title = "Introduction"
        current_elements: list[ParsedElement] = []

        for el in elements:
            if el.is_structural_boundary():
                if current_elements:
                    sections.append(
                        {
                            "title": current_title,
                            "elements": current_elements,
                            "atomic": None,
                        }
                    )
                    current_elements = []
                current_title = el.content.strip("# ").strip()

            elif el.is_atomic():
                if current_elements:
                    sections.append(
                        {
                            "title": current_title,
                            "elements": current_elements,
                            "atomic": None,
                        }
                    )
                    current_elements = []
                sections.append(
                    {
                        "title": current_title,
                        "elements": [],
                        "atomic": el,
                    }
                )

            else:
                current_elements.append(el)

        if current_elements:
            sections.append(
                {
                    "title": current_title,
                    "elements": current_elements,
                    "atomic": None,
                }
            )

        return sections

    def _merge_small_sections(self, sections: list[dict]) -> list[dict]:
        """Merge consecutive text sections that are below parent_min_words."""
        if not sections:
            return sections

        merged: list[dict] = []
        buffer: dict | None = None

        for section in sections:
            # Atomic sections are never merged
            if section["atomic"]:
                if buffer:
                    merged.append(buffer)
                    buffer = None
                merged.append(section)
                continue

            if buffer is None:
                buffer = {
                    "title": section["title"],
                    "elements": list(section["elements"]),
                    "atomic": None,
                }
            else:
                # Merge into buffer
                buffer["elements"].extend(section["elements"])

            buffer_words = sum(
                self._count_words(el.content) for el in buffer["elements"]
            )
            if buffer_words >= self.parent_min_words:
                merged.append(buffer)
                buffer = None

        if buffer:
            # If buffer is still small, merge with last text section if possible
            if merged and not merged[-1]["atomic"]:
                merged[-1]["elements"].extend(buffer["elements"])
            else:
                merged.append(buffer)

        return merged

    # ── parent + child creation ─────────────────────────────────

    def _build_parent_children(
        self,
        section: dict,
        doc_metadata: dict,
    ) -> tuple[list[Chunk], list[Chunk]]:
        """Build parents and children from a text section."""
        elements = section["elements"]
        if not elements:
            return [], []

        title = section["title"]
        full_text = "\n\n".join(el.content for el in elements)
        paragraphs = self._extract_paragraphs(full_text)

        # Split into parent-sized groups of paragraphs
        parent_groups = self._group_paragraphs_into_parents(paragraphs)

        all_parents: list[Chunk] = []
        all_children: list[Chunk] = []
        prev_last_sentence: str | None = None

        first_el = elements[0]
        for group in parent_groups:
            parent_text = "\n\n".join(group)
            parent_id = str(uuid.uuid4())

            parent = Chunk(
                chunk_id=parent_id,
                parent_id=None,
                doc_id=first_el.doc_id,
                doc_name=first_el.doc_name,
                content=parent_text,
                content_raw=parent_text,
                type="text",
                page=first_el.page,
                section=title,
                language=first_el.language or "en",
                word_count=self._count_words(parent_text),
                is_parent=True,
                metadata=doc_metadata,
            )
            all_parents.append(parent)

            # Build children from paragraphs in this parent
            child_paragraphs = self._build_child_paragraphs(group)
            for i, child_text in enumerate(child_paragraphs):
                # Sentence overlap from previous parent group's last child
                content_with_overlap = child_text
                overlap_sentence = None
                if prev_last_sentence is not None and i == 0 and len(all_parents) > 1:
                    overlap_sentence = prev_last_sentence
                    content_with_overlap = f"{prev_last_sentence}\n{child_text}"

                child = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    parent_id=parent_id,
                    doc_id=first_el.doc_id,
                    doc_name=first_el.doc_name,
                    content=content_with_overlap,
                    content_raw=child_text,
                    type="text",
                    page=first_el.page,
                    section=title,
                    language=first_el.language or "en",
                    word_count=self._count_words(content_with_overlap),
                    is_parent=False,
                    metadata={
                        **doc_metadata,
                        "chunk_index": i,
                        **(
                            {"overlap_sentence": overlap_sentence}
                            if overlap_sentence
                            else {}
                        ),
                    },
                )
                all_children.append(child)

            # Track last sentence for overlap into next parent group
            if child_paragraphs:
                last_child = child_paragraphs[-1]
                sents = split_sentences(last_child)
                prev_last_sentence = sents[-1] if sents else None

        return all_parents, all_children

    # ── paragraph extraction + grouping ─────────────────────────

    def _extract_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs by double newlines."""
        raw = text.split("\n\n")
        return [p.strip() for p in raw if p.strip()]

    def _group_paragraphs_into_parents(
        self,
        paragraphs: list[str],
    ) -> list[list[str]]:
        """Group paragraphs into parent-sized groups respecting max_words."""
        if not paragraphs:
            return []

        groups: list[list[str]] = []
        current_group: list[str] = []
        current_words = 0

        for para in paragraphs:
            para_words = self._count_words(para)

            if current_words + para_words > self.parent_max_words and current_group:
                groups.append(current_group)
                current_group = []
                current_words = 0

            current_group.append(para)
            current_words += para_words

        if current_group:
            groups.append(current_group)

        return groups

    def _build_child_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """
        Build child chunks from paragraphs:
        - Merge small paragraphs (< child_min_words) with next
        - Split large paragraphs (> child_max_words) at sentence boundaries
        """
        result: list[str] = []
        buffer = ""
        buffer_words = 0

        for para in paragraphs:
            para_words = self._count_words(para)

            if para_words > self.child_max_words:
                # Flush buffer first
                if buffer:
                    result.append(buffer.strip())
                    buffer = ""
                    buffer_words = 0
                # Split oversized paragraph at sentence boundaries
                result.extend(self._split_at_sentences(para))

            elif buffer_words + para_words < self.child_min_words:
                # Accumulate small paragraphs
                buffer = f"{buffer}\n\n{para}" if buffer else para
                buffer_words += para_words

            else:
                # Flush accumulated buffer + current para if combined is in range
                if buffer:
                    combined = f"{buffer}\n\n{para}"
                    combined_words = buffer_words + para_words
                    if combined_words <= self.child_max_words:
                        result.append(combined.strip())
                        buffer = ""
                        buffer_words = 0
                    else:
                        result.append(buffer.strip())
                        result.append(para.strip())
                        buffer = ""
                        buffer_words = 0
                else:
                    result.append(para.strip())

        if buffer:
            # Merge trailing small buffer with last result if possible
            if result:
                last = result[-1]
                combined = f"{last}\n\n{buffer}"
                if self._count_words(combined) <= self.child_max_words:
                    result[-1] = combined.strip()
                else:
                    result.append(buffer.strip())
            else:
                result.append(buffer.strip())

        return result

    def _split_at_sentences(self, text: str) -> list[str]:
        """Split oversized text at sentence boundaries targeting child_target words."""
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return [text]

        chunks: list[str] = []
        current: list[str] = []
        current_words = 0

        for sent in sentences:
            sent_words = self._count_words(sent)
            if current_words + sent_words > self.child_max_words and current:
                chunks.append(" ".join(current))
                current = []
                current_words = 0
            current.append(sent)
            current_words += sent_words

        if current:
            chunks.append(" ".join(current))

        return chunks

    # ── atomic chunks (unchanged) ───────────────────────────────

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

    @staticmethod
    def _count_words(text: str) -> int:
        return len(text.split())
