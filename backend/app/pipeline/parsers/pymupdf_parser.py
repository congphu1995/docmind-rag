import time

import pymupdf4llm

from backend.app.core.exceptions import ParserError
from backend.app.core.logging import logger
from backend.app.pipeline.base.parser import (
    BaseParser,
    ParsedElement,
    ParserCapabilities,
)
from backend.app.pipeline.parsers.normalizer import ElementNormalizer


class PyMuPDFParser(BaseParser):
    def __init__(self):
        self._normalizer = ElementNormalizer()

    def supports(self, file_ext: str) -> bool:
        return file_ext.lower() == ".pdf"

    def get_capabilities(self) -> ParserCapabilities:
        return ParserCapabilities(
            handles_scanned=False,
            handles_tables=True,
            handles_figures=False,
            output_confidence=False,
        )

    async def parse(
        self, file_path: str, doc_id: str, doc_name: str, **kwargs
    ) -> list[ParsedElement]:
        start = time.perf_counter()
        logger.info("pymupdf_parse_start", doc_id=doc_id, file=file_path)

        try:
            pages_data = pymupdf4llm.to_markdown(file_path, page_chunks=True)
            elements = self._normalizer.from_pymupdf(pages_data, doc_id, doc_name)

            elapsed = time.perf_counter() - start
            logger.info(
                "pymupdf_parse_done",
                doc_id=doc_id,
                elements=len(elements),
                elapsed_s=round(elapsed, 2),
            )
            return elements

        except Exception as e:
            logger.error("pymupdf_parse_failed", doc_id=doc_id, error=str(e))
            raise ParserError(f"PyMuPDF failed on {file_path}: {e}") from e
