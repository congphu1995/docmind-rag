from pathlib import Path

from backend.app.core.config import settings
from backend.app.pipeline.base.parser import BaseParser
from backend.app.pipeline.parsers.docling_parser import DoclingParser
from backend.app.pipeline.parsers.preprocessor import PDFPreprocessor
from backend.app.pipeline.parsers.pymupdf_parser import PyMuPDFParser


class ParserFactory:
    @staticmethod
    def create(strategy: str = None) -> BaseParser:
        """Create parser by explicit strategy name."""
        strategy = strategy or settings.parser_strategy
        parsers = {
            "docling": DoclingParser,
            "pymupdf": PyMuPDFParser,
        }
        if strategy not in parsers and strategy != "auto":
            raise ValueError(
                f"Unknown parser strategy: {strategy}. Choose: {list(parsers.keys())}"
            )
        if strategy == "auto":
            return DoclingParser()
        return parsers[strategy]()

    @staticmethod
    def auto_select(file_path: str) -> BaseParser:
        """
        Select best parser based on file type + content analysis.
        Clean PDF → PyMuPDF (fast)
        Scanned PDF → Docling (OCR)
        DOCX / TXT → Docling
        """
        ext = Path(file_path).suffix.lower()

        if ext != ".pdf":
            return DoclingParser()

        diagnosis = PDFPreprocessor().diagnose(file_path)

        if diagnosis["is_scanned"]:
            return DoclingParser(enable_ocr=True)

        if diagnosis["page_count"] > 50:
            return PyMuPDFParser()

        return DoclingParser(enable_ocr=False)
