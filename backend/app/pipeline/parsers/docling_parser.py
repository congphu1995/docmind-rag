import time

from docling.document_converter import DocumentConverter

from backend.app.core.exceptions import ParserError
from backend.app.core.logging import logger
from backend.app.pipeline.base.parser import (
    BaseParser,
    ParsedElement,
    ParserCapabilities,
)
from backend.app.pipeline.parsers.normalizer import ElementNormalizer


class DoclingParser(BaseParser):

    def __init__(
        self, enable_ocr: bool = True, enable_table_structure: bool = True
    ):
        self._converter = DocumentConverter()
        self._normalizer = ElementNormalizer()
        self._enable_ocr = enable_ocr

    def supports(self, file_ext: str) -> bool:
        return file_ext.lower() in {".pdf", ".docx", ".txt", ".md", ".html"}

    def get_capabilities(self) -> ParserCapabilities:
        return ParserCapabilities(
            handles_scanned=self._enable_ocr,
            handles_tables=True,
            handles_figures=True,
            output_confidence=False,
        )

    async def parse(
        self, file_path: str, doc_id: str, doc_name: str, **kwargs
    ) -> list[ParsedElement]:
        start = time.perf_counter()
        logger.info("docling_parse_start", doc_id=doc_id, file=file_path)

        try:
            result = self._converter.convert(file_path)
            elements = self._normalizer.from_docling(result, doc_id, doc_name)

            elapsed = time.perf_counter() - start
            logger.info(
                "docling_parse_done",
                doc_id=doc_id,
                elements=len(elements),
                elapsed_s=round(elapsed, 2),
            )
            return elements

        except Exception as e:
            logger.error("docling_parse_failed", doc_id=doc_id, error=str(e))
            raise ParserError(f"Docling failed on {file_path}: {e}") from e
