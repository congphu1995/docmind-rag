"""
Routes each ParsedElement to the correct chunking strategy.
Entry point for all chunking — nothing calls ParentChildChunker directly.
"""
from __future__ import annotations

from backend.app.core.logging import logger
from backend.app.pipeline.base.chunker import Chunk
from backend.app.pipeline.base.parser import ElementType, ParsedElement
from backend.app.pipeline.chunkers.parent_child_chunker import ParentChildChunker

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.pipeline.multimodal.figure_describer import FigureDescriber
    from backend.app.pipeline.multimodal.table_representer import TableRepresenter


class SmartRouter:

    def __init__(
        self,
        chunker: ParentChildChunker = None,
        figure_describer: FigureDescriber = None,
        table_representer: TableRepresenter = None,
    ):
        self._chunker = chunker or ParentChildChunker()
        self._figure_describer = figure_describer
        self._table_representer = table_representer

    async def route(
        self,
        elements: list[ParsedElement],
        doc_metadata: dict,
    ) -> tuple[list[Chunk], list[Chunk]]:
        elements = self._group_list_items(elements)

        # Enrich figures and tables with multimodal processing
        if self._figure_describer or self._table_representer:
            elements = await self._enrich_multimodal(elements, doc_metadata)

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

    async def _enrich_multimodal(
        self,
        elements: list[ParsedElement],
        doc_metadata: dict,
    ) -> list[ParsedElement]:
        """Add text descriptions to figures and NL to tables."""
        doc_context = (
            f"{doc_metadata.get('doc_name', '')} — "
            f"{doc_metadata.get('doc_type', 'document')}"
        )

        for el in elements:
            if (
                el.type == ElementType.FIGURE
                and el.image_b64
                and self._figure_describer
            ):
                section_ctx = f"{doc_context}, {el.section_title or 'unknown section'}"
                description = await self._figure_describer.describe(
                    image_b64=el.image_b64,
                    doc_context=section_ctx,
                )
                el.content = description
                logger.info("figure_enriched", page=el.page)

            if el.type == ElementType.TABLE and self._table_representer:
                html = el.table_html or ""
                if html:
                    reps = await self._table_representer.represent(
                        table_html=html,
                        section_context=el.section_title or "",
                    )
                    # content = NL (for embedding)
                    el.content = reps.natural_language
                    # Keep markdown + html for downstream
                    el.table_html = reps.html
                    logger.info("table_enriched", page=el.page)

        return elements

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
