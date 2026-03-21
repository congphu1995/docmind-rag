from backend.app.pipeline.base.parser import ParsedElement, ElementType
from backend.app.pipeline.base.chunker import Chunk


def test_parsed_element_is_atomic_table():
    table = ParsedElement(
        type=ElementType.TABLE, content="...", page=0, doc_id="x", doc_name="x"
    )
    assert table.is_atomic() is True


def test_parsed_element_is_atomic_text():
    text = ParsedElement(
        type=ElementType.TEXT, content="...", page=0, doc_id="x", doc_name="x"
    )
    assert text.is_atomic() is False


def test_parsed_element_is_atomic_figure():
    fig = ParsedElement(
        type=ElementType.FIGURE, content="...", page=0, doc_id="x", doc_name="x"
    )
    assert fig.is_atomic() is True


def test_parsed_element_is_atomic_code():
    code = ParsedElement(
        type=ElementType.CODE, content="...", page=0, doc_id="x", doc_name="x"
    )
    assert code.is_atomic() is True


def test_parsed_element_is_structural_boundary():
    title = ParsedElement(
        type=ElementType.TITLE,
        content="Section 1",
        page=0,
        doc_id="x",
        doc_name="x",
    )
    assert title.is_structural_boundary() is True

    text = ParsedElement(
        type=ElementType.TEXT, content="hello", page=0, doc_id="x", doc_name="x"
    )
    assert text.is_structural_boundary() is False


def test_parsed_element_word_count():
    el = ParsedElement(
        type=ElementType.TEXT,
        content="one two three four",
        page=0,
        doc_id="x",
        doc_name="x",
    )
    assert el.word_count() == 4


def test_chunk_to_document_includes_all_fields():
    chunk = Chunk(
        doc_id="abc",
        content="enriched text",
        content_raw="raw text",
        content_html="<p>raw text</p>",
        page=1,
        is_parent=False,
        user_id="user1",
        metadata={"doc_type": "report"},
    )
    doc = chunk.to_document()
    assert isinstance(doc, dict)
    assert doc["doc_id"] == "abc"
    assert doc["content"] == "enriched text"
    assert doc["content_raw"] == "raw text"
    assert doc["content_html"] == "<p>raw text</p>"
    assert doc["user_id"] == "user1"
    assert doc["is_parent"] is False
    assert doc["metadata"] == {"doc_type": "report"}
    assert "created_at" in doc


def test_chunk_to_document_metadata_is_nested():
    """Metadata must be a nested dict, not flattened into top-level keys."""
    chunk = Chunk(
        doc_id="abc",
        content="test",
        content_raw="test",
        metadata={"doc_type": "report", "date": "2024-01-01"},
    )
    doc = chunk.to_document()
    assert "metadata" in doc
    assert doc["metadata"]["doc_type"] == "report"
    assert "doc_type" not in doc


def test_chunk_default_user_id():
    chunk = Chunk()
    assert chunk.user_id == ""


def test_chunk_default_values():
    chunk = Chunk()
    assert chunk.chunk_id  # UUID generated
    assert chunk.parent_id is None
    assert chunk.is_parent is False
    assert chunk.type == "text"
