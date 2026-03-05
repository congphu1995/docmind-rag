"""
Routes each ParsedElement to the correct chunking strategy.
Entry point for all chunking — nothing calls ParentChildChunker directly.
"""
from backend.app.core.logging import logger
from backend.app.pipeline.base.chunker import Chunk
from backend.app.pipeline.base.parser import ElementType, ParsedElement
from backend.app.pipeline.chunkers.parent_child_chunker import ParentChildChunker


class SmartRouter:

    def __init__(self, chunker: ParentChildChunker = None):
        self._chunker = chunker or ParentChildChunker()

    def route(
        self,
        elements: list[ParsedElement],
        doc_metadata: dict,
    ) -> tuple[list[Chunk], list[Chunk]]:
        elements = self._group_list_items(elements)

        type_counts: dict[str, int] = {}
        for el in elements:
            type_counts[el.type.value] = type_counts.get(el.type.value, 0) + 1
        logger.info("chunking_routing", **type_counts)

        parents, children = self._chunker.chunk(elements, doc_metadata)

        logger.info(
            "chunking_done",
            doc_id=doc_metadata.get("doc_id", ""),
            parents=len(parents),
            children=len(children),
        )
        return parents, children

    def _group_list_items(
        self, elements: list[ParsedElement]
    ) -> list[ParsedElement]:
        """Merge consecutive list items into one TEXT element."""
        result = []
        list_buffer: list[ParsedElement] = []

        for el in elements:
            if el.type == ElementType.LIST_ITEM:
                list_buffer.append(el)
            else:
                if list_buffer:
                    merged = self._merge_list_items(list_buffer)
                    result.append(merged)
                    list_buffer = []
                result.append(el)

        if list_buffer:
            result.append(self._merge_list_items(list_buffer))

        return result

    def _merge_list_items(
        self, items: list[ParsedElement]
    ) -> ParsedElement:
        merged_content = "\n".join(f"• {el.content.strip()}" for el in items)
        first = items[0]
        return ParsedElement(
            type=ElementType.TEXT,
            content=merged_content,
            page=first.page,
            doc_id=first.doc_id,
            doc_name=first.doc_name,
            section_title=first.section_title,
            language=first.language,
            parser_used=first.parser_used,
        )
