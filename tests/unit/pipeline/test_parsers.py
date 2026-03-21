import pytest
from unittest.mock import patch

from backend.app.pipeline.parsers.factory import ParserFactory


def test_factory_creates_docling():
    parser = ParserFactory.create("docling")
    from backend.app.pipeline.parsers.docling_parser import DoclingParser

    assert isinstance(parser, DoclingParser)


def test_factory_creates_pymupdf():
    parser = ParserFactory.create("pymupdf")
    from backend.app.pipeline.parsers.pymupdf_parser import PyMuPDFParser

    assert isinstance(parser, PyMuPDFParser)


def test_factory_raises_on_unknown():
    with pytest.raises(ValueError, match="Unknown parser strategy"):
        ParserFactory.create("nonexistent")


def test_pymupdf_supports_only_pdf():
    from backend.app.pipeline.parsers.pymupdf_parser import PyMuPDFParser

    p = PyMuPDFParser()
    assert p.supports(".pdf") is True
    assert p.supports(".docx") is False


def test_docling_supports_multiple_formats():
    from backend.app.pipeline.parsers.docling_parser import DoclingParser

    p = DoclingParser()
    for ext in [".pdf", ".docx", ".txt", ".md"]:
        assert p.supports(ext) is True


@patch("backend.app.pipeline.parsers.factory.PDFPreprocessor")
def test_auto_select_scanned_returns_docling(mock_preprocessor):
    mock_preprocessor.return_value.diagnose.return_value = {
        "is_scanned": True,
        "page_count": 5,
    }
    parser = ParserFactory.auto_select("scanned.pdf")
    from backend.app.pipeline.parsers.docling_parser import DoclingParser

    assert isinstance(parser, DoclingParser)


@patch("backend.app.pipeline.parsers.factory.PDFPreprocessor")
def test_auto_select_large_clean_returns_pymupdf(mock_preprocessor):
    mock_preprocessor.return_value.diagnose.return_value = {
        "is_scanned": False,
        "page_count": 100,
    }
    parser = ParserFactory.auto_select("large.pdf")
    from backend.app.pipeline.parsers.pymupdf_parser import PyMuPDFParser

    assert isinstance(parser, PyMuPDFParser)


def test_auto_select_docx_returns_docling():
    parser = ParserFactory.auto_select("document.docx")
    from backend.app.pipeline.parsers.docling_parser import DoclingParser

    assert isinstance(parser, DoclingParser)
