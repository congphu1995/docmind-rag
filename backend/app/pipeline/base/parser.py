from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    TITLE = "title"
    LIST_ITEM = "list_item"
    CODE = "code"
    SCANNED = "scanned"
    CAPTION = "caption"


@dataclass
class ParsedElement:
    type: ElementType
    content: str
    page: int
    doc_id: str
    doc_name: str

    section_title: Optional[str] = None
    level: Optional[int] = None
    reading_order: Optional[int] = None
    bbox: Optional[tuple] = None

    image_b64: Optional[str] = None
    table_html: Optional[str] = None
    table_df: Optional[pd.DataFrame] = None

    confidence: Optional[float] = None
    language: Optional[str] = None
    is_scanned: bool = False

    parser_used: str = ""

    def is_atomic(self) -> bool:
        """Tables, figures, code are never split."""
        return self.type in (ElementType.TABLE, ElementType.FIGURE, ElementType.CODE)

    def is_structural_boundary(self) -> bool:
        """Titles mark section boundaries, not chunks themselves."""
        return self.type == ElementType.TITLE

    def word_count(self) -> int:
        return len(self.content.split())


@dataclass
class ParserCapabilities:
    handles_scanned: bool = False
    handles_tables: bool = True
    handles_figures: bool = False
    max_pages: Optional[int] = None
    output_confidence: bool = False


class BaseParser(ABC):
    """
    Abstract base — all parsers implement this interface.
    Services import BaseParser only, never a concrete class.
    """

    @abstractmethod
    async def parse(
        self,
        file_path: str,
        doc_id: str,
        doc_name: str,
        **kwargs,
    ) -> list[ParsedElement]:
        ...

    @abstractmethod
    def supports(self, file_ext: str) -> bool:
        ...

    @abstractmethod
    def get_capabilities(self) -> ParserCapabilities:
        ...
