from pathlib import Path

import fitz

from backend.app.core.exceptions import EncryptedDocumentError
from backend.app.core.logging import logger


class PDFPreprocessor:

    def diagnose(self, file_path: str) -> dict:
        """Inspect PDF before parsing. Returns diagnosis dict."""
        doc = fitz.open(file_path)
        page = doc[0] if doc.page_count > 0 else None

        text_len = sum(len(p.get_text()) for p in doc) if doc.page_count > 0 else 0
        page_area = (page.rect.width * page.rect.height) if page else 1

        return {
            "is_scanned": text_len < 50 and doc.page_count > 0,
            "is_rotated": page.rotation != 0 if page else False,
            "is_encrypted": doc.is_encrypted,
            "is_corrupt": doc.is_repaired,
            "page_count": doc.page_count,
            "text_density": text_len / (page_area * doc.page_count + 1),
            "file_size_mb": Path(file_path).stat().st_size / (1024 * 1024),
        }

    def preprocess(self, file_path: str) -> str:
        """
        Run diagnosis and fix issues.
        Returns path to cleaned file (may be same as input).
        """
        diagnosis = self.diagnose(file_path)
        logger.info("pdf_diagnosis", file=file_path, **diagnosis)

        if diagnosis["is_encrypted"]:
            raise EncryptedDocumentError(
                f"Document is password-protected: {file_path}. "
                "Please provide an unencrypted version."
            )

        if diagnosis["is_rotated"]:
            file_path = self._deskew(file_path)
            logger.info("pdf_deskewed", file=file_path)

        return file_path

    def _deskew(self, file_path: str) -> str:
        """Normalize page rotation."""
        doc = fitz.open(file_path)
        out_path = file_path.replace(".pdf", "_deskewed.pdf")
        for page in doc:
            if page.rotation != 0:
                page.set_rotation(0)
        doc.save(out_path)
        return out_path
