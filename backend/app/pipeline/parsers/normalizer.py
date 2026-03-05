"""
Converts raw parser output (varies by parser) → unified ParsedElement list.
All downstream code only ever sees ParsedElement.
"""
from typing import Optional

from backend.app.pipeline.base.parser import ElementType, ParsedElement


class ElementNormalizer:

    def from_docling(
        self, docling_result, doc_id: str, doc_name: str
    ) -> list[ParsedElement]:
        """Convert Docling Document object → List[ParsedElement]."""
        elements = []
        current_section = ""

        for item in docling_result.document.body.children:
            el_type = self._map_docling_type(item)
            content = self._extract_docling_content(item)

            if not content or not content.strip():
                continue

            if el_type == ElementType.TITLE:
                current_section = content.strip()

            elements.append(
                ParsedElement(
                    type=el_type,
                    content=content,
                    page=self._get_docling_page(item),
                    doc_id=doc_id,
                    doc_name=doc_name,
                    section_title=(
                        current_section if el_type != ElementType.TITLE else None
                    ),
                    table_html=(
                        self._get_table_html(item)
                        if el_type == ElementType.TABLE
                        else None
                    ),
                    parser_used="docling",
                )
            )

        return self._assign_reading_order(elements)

    def from_pymupdf(
        self, pages_data: list[dict], doc_id: str, doc_name: str
    ) -> list[ParsedElement]:
        """Convert pymupdf4llm page dict list → List[ParsedElement]."""
        elements = []
        current_section = ""

        for page_data in pages_data:
            page_num = page_data.get("metadata", {}).get("page", 0)
            md_content = page_data.get("text", "")

            for block in self._parse_markdown_blocks(md_content):
                el_type = block["type"]
                content = block["content"]

                if not content.strip():
                    continue

                if el_type == ElementType.TITLE:
                    current_section = content.strip("# ").strip()

                elements.append(
                    ParsedElement(
                        type=el_type,
                        content=content,
                        page=page_num,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        section_title=current_section,
                        parser_used="pymupdf",
                    )
                )

        return self._assign_reading_order(elements)

    def _map_docling_type(self, item) -> ElementType:
        type_map = {
            "section_header": ElementType.TITLE,
            "text": ElementType.TEXT,
            "table": ElementType.TABLE,
            "figure": ElementType.FIGURE,
            "list_item": ElementType.LIST_ITEM,
            "code": ElementType.CODE,
        }
        item_type = getattr(item, "label", "text")
        return type_map.get(str(item_type).lower(), ElementType.TEXT)

    def _parse_markdown_blocks(self, md: str) -> list[dict]:
        """Parse markdown string into typed blocks."""
        blocks = []
        current_table: list[str] = []
        in_table = False
        lines = md.split("\n")

        for line in lines:
            if line.startswith("#"):
                if in_table and current_table:
                    blocks.append(
                        {"type": ElementType.TABLE, "content": "\n".join(current_table)}
                    )
                    current_table = []
                    in_table = False
                blocks.append({"type": ElementType.TITLE, "content": line})
            elif line.startswith("|"):
                in_table = True
                current_table.append(line)
            elif in_table and not line.startswith("|"):
                blocks.append(
                    {"type": ElementType.TABLE, "content": "\n".join(current_table)}
                )
                current_table = []
                in_table = False
                if line.strip():
                    blocks.append({"type": ElementType.TEXT, "content": line})
            elif line.strip():
                blocks.append({"type": ElementType.TEXT, "content": line})

        if current_table:
            blocks.append(
                {"type": ElementType.TABLE, "content": "\n".join(current_table)}
            )

        return blocks

    def _get_docling_page(self, item) -> int:
        try:
            return item.prov[0].page_no - 1
        except (AttributeError, IndexError):
            return 0

    def _get_table_html(self, item) -> Optional[str]:
        try:
            return item.export_to_html()
        except Exception:
            return None

    def _extract_docling_content(self, item) -> str:
        try:
            return item.export_to_markdown()
        except Exception:
            return str(getattr(item, "text", ""))

    def _assign_reading_order(
        self, elements: list[ParsedElement]
    ) -> list[ParsedElement]:
        for i, el in enumerate(elements):
            el.reading_order = i
        return elements
