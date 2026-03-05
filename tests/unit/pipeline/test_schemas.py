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


def test_chunk_qdrant_payload_is_flat():
    chunk = Chunk(doc_id="abc", content="test", content_raw="test", page=1)
    payload = chunk.qdrant_payload()
    assert isinstance(payload, dict)
    assert "doc_id" in payload
    assert "content_raw" in payload
    assert all(not isinstance(v, dict) for v in payload.values())


def test_chunk_qdrant_payload_includes_metadata():
    chunk = Chunk(
        doc_id="abc",
        content="test",
        content_raw="test",
        metadata={"doc_type": "report", "date": "2024-01-01"},
    )
    payload = chunk.qdrant_payload()
    assert payload["doc_type"] == "report"
    assert payload["date"] == "2024-01-01"


def test_chunk_default_values():
    chunk = Chunk()
    assert chunk.chunk_id  # UUID generated
    assert chunk.parent_id is None
    assert chunk.is_parent is False
    assert chunk.type == "text"
