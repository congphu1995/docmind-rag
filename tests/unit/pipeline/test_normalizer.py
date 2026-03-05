from backend.app.pipeline.parsers.normalizer import ElementNormalizer
from backend.app.pipeline.base.parser import ElementType


def test_parse_markdown_blocks_title():
    normalizer = ElementNormalizer()
    blocks = normalizer._parse_markdown_blocks("# Section 1\n\nSome text.")
    types = [b["type"] for b in blocks]
    assert ElementType.TITLE in types
    assert ElementType.TEXT in types


def test_parse_markdown_blocks_table():
    normalizer = ElementNormalizer()
    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    blocks = normalizer._parse_markdown_blocks(md)
    assert any(b["type"] == ElementType.TABLE for b in blocks)


def test_parse_markdown_blocks_mixed():
    normalizer = ElementNormalizer()
    md = "# Section 1\n\nSome text here.\n\n| A | B |\n|---|---|\n| 1 | 2 |"
    blocks = normalizer._parse_markdown_blocks(md)
    types = [b["type"] for b in blocks]
    assert ElementType.TITLE in types
    assert ElementType.TEXT in types
    assert ElementType.TABLE in types


def test_from_pymupdf_sets_parser_used():
    normalizer = ElementNormalizer()
    pages_data = [{"metadata": {"page": 0}, "text": "# Title\n\nSome text."}]
    elements = normalizer.from_pymupdf(pages_data, "doc1", "test.pdf")
    assert all(el.parser_used == "pymupdf" for el in elements)


def test_from_pymupdf_tracks_sections():
    normalizer = ElementNormalizer()
    pages_data = [
        {"metadata": {"page": 0}, "text": "# Section A\n\nContent under A."}
    ]
    elements = normalizer.from_pymupdf(pages_data, "doc1", "test.pdf")
    text_elements = [el for el in elements if el.type == ElementType.TEXT]
    assert all(el.section_title == "Section A" for el in text_elements)


def test_empty_content_filtered():
    normalizer = ElementNormalizer()
    pages_data = [{"metadata": {"page": 0}, "text": "\n\n\n"}]
    elements = normalizer.from_pymupdf(pages_data, "doc1", "test.pdf")
    assert len(elements) == 0


def test_assign_reading_order():
    normalizer = ElementNormalizer()
    pages_data = [
        {"metadata": {"page": 0}, "text": "# Title\n\nParagraph one.\n\nParagraph two."}
    ]
    elements = normalizer.from_pymupdf(pages_data, "doc1", "test.pdf")
    for i, el in enumerate(elements):
        assert el.reading_order == i
