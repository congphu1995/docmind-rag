import pytest
from unittest.mock import patch, MagicMock

from backend.app.pipeline.parsers.preprocessor import PDFPreprocessor
from backend.app.core.exceptions import EncryptedDocumentError


def _mock_fitz_doc(
    page_count=5,
    text_len=500,
    rotation=0,
    is_encrypted=False,
    is_repaired=False,
    width=612,
    height=792,
):
    """Create a mock fitz document."""
    mock_page = MagicMock()
    mock_page.get_text.return_value = "x" * (text_len // max(page_count, 1))
    mock_page.rotation = rotation
    mock_page.rect.width = width
    mock_page.rect.height = height

    mock_doc = MagicMock()
    mock_doc.page_count = page_count
    mock_doc.is_encrypted = is_encrypted
    mock_doc.is_repaired = is_repaired
    mock_doc.__iter__ = lambda self: iter([mock_page] * page_count)
    mock_doc.__getitem__ = lambda self, i: mock_page
    return mock_doc


@patch("backend.app.pipeline.parsers.preprocessor.fitz.open")
@patch("backend.app.pipeline.parsers.preprocessor.Path")
def test_diagnose_normal_pdf(mock_path, mock_fitz_open):
    mock_fitz_open.return_value = _mock_fitz_doc()
    mock_path.return_value.stat.return_value.st_size = 1024 * 1024

    preprocessor = PDFPreprocessor()
    result = preprocessor.diagnose("test.pdf")

    assert result["is_scanned"] is False
    assert result["is_encrypted"] is False
    assert result["is_rotated"] is False
    assert result["page_count"] == 5


@patch("backend.app.pipeline.parsers.preprocessor.fitz.open")
@patch("backend.app.pipeline.parsers.preprocessor.Path")
def test_diagnose_scanned_pdf(mock_path, mock_fitz_open):
    mock_fitz_open.return_value = _mock_fitz_doc(text_len=10)
    mock_path.return_value.stat.return_value.st_size = 1024 * 1024

    preprocessor = PDFPreprocessor()
    result = preprocessor.diagnose("scanned.pdf")

    assert result["is_scanned"] is True


@patch("backend.app.pipeline.parsers.preprocessor.fitz.open")
@patch("backend.app.pipeline.parsers.preprocessor.Path")
def test_diagnose_rotated_pdf(mock_path, mock_fitz_open):
    mock_fitz_open.return_value = _mock_fitz_doc(rotation=90)
    mock_path.return_value.stat.return_value.st_size = 1024 * 1024

    preprocessor = PDFPreprocessor()
    result = preprocessor.diagnose("rotated.pdf")

    assert result["is_rotated"] is True


@patch("backend.app.pipeline.parsers.preprocessor.fitz.open")
@patch("backend.app.pipeline.parsers.preprocessor.Path")
def test_preprocess_encrypted_raises(mock_path, mock_fitz_open):
    mock_fitz_open.return_value = _mock_fitz_doc(is_encrypted=True)
    mock_path.return_value.stat.return_value.st_size = 1024 * 1024

    preprocessor = PDFPreprocessor()
    with pytest.raises(EncryptedDocumentError):
        preprocessor.preprocess("encrypted.pdf")
