from backend.app.pipeline.base.chunker import Chunk
from backend.app.pipeline.base.parser import ElementType, ParsedElement
from backend.app.pipeline.chunkers.parent_child_chunker import ParentChildChunker
from backend.app.pipeline.chunkers.quality_filter import QualityFilter
from backend.app.pipeline.chunkers.smart_router import SmartRouter


def make_element(
    type: ElementType, content: str, page: int = 0
) -> ParsedElement:
    return ParsedElement(
        type=type, content=content, page=page, doc_id="test", doc_name="test.pdf"
    )


def test_tables_are_always_atomic():
    chunker = ParentChildChunker()
    table_el = make_element(ElementType.TABLE, "| A | B |\n|---|---|\n| 1 | 2 |")
    parents, children = chunker.chunk([table_el], {})

    assert len(parents) == 1
    assert len(children) == 1
    assert parents[0].chunk_id == children[0].chunk_id
    assert parents[0].type == "table"


def test_title_creates_section_boundary_not_chunk():
    chunker = ParentChildChunker()
    elements = [
        make_element(ElementType.TITLE, "# Section 1"),
        make_element(ElementType.TEXT, "Some content here " * 20),
    ]
    parents, children = chunker.chunk(elements, {})

    all_chunks = parents + children
    assert not any(c.type == "title" for c in all_chunks)
    assert all(c.section == "Section 1" for c in children)


def test_children_have_parent_id():
    chunker = ParentChildChunker(parent_max_words=50, child_min_words=5, child_max_words=15)
    long_text = "word " * 200
    elements = [make_element(ElementType.TEXT, long_text)]
    parents, children = chunker.chunk(elements, {})

    parent_ids = {p.chunk_id for p in parents if p.is_parent}
    for child in children:
        if not child.is_parent:
            assert child.parent_id in parent_ids


def test_short_text_produces_single_child():
    chunker = ParentChildChunker(parent_max_words=800, child_max_words=150)
    elements = [make_element(ElementType.TEXT, "Short text.")]
    parents, children = chunker.chunk(elements, {})

    assert len(parents) == 1
    assert len(children) == 1
    assert children[0].parent_id == parents[0].chunk_id


def test_quality_filter_removes_short_chunks():
    filt = QualityFilter()
    short = Chunk(
        content_raw="too short", content="too short", type="text", is_parent=False
    )
    long_enough = Chunk(
        content_raw="word " * 20, content="word " * 20, type="text", is_parent=False
    )
    result = filt.filter([short, long_enough])
    assert short not in result
    assert long_enough in result


def test_quality_filter_keeps_tables():
    filt = QualityFilter()
    table = Chunk(content_raw="1", content="1", type="table", is_parent=True)
    result = filt.filter([table])
    assert table in result


def test_quality_filter_keeps_parents():
    filt = QualityFilter()
    parent = Chunk(
        content_raw="short", content="short", type="text", is_parent=True
    )
    result = filt.filter([parent])
    assert parent in result


def test_quality_filter_removes_repeated_chars():
    filt = QualityFilter()
    noisy = Chunk(
        content_raw="aaaaaa " * 20,
        content="aaaaaa " * 20,
        type="text",
        is_parent=False,
    )
    result = filt.filter([noisy])
    assert noisy not in result


def test_smart_router_groups_list_items():
    router = SmartRouter()
    elements = [
        make_element(ElementType.LIST_ITEM, "Item one"),
        make_element(ElementType.LIST_ITEM, "Item two"),
        make_element(ElementType.LIST_ITEM, "Item three"),
    ]
    grouped = router._group_list_items(elements)
    assert len(grouped) == 1
    assert grouped[0].type == ElementType.TEXT
    assert "Item one" in grouped[0].content
    assert "Item two" in grouped[0].content


def test_smart_router_preserves_non_list_items():
    router = SmartRouter()
    elements = [
        make_element(ElementType.TEXT, "Regular text"),
        make_element(ElementType.LIST_ITEM, "Item one"),
        make_element(ElementType.LIST_ITEM, "Item two"),
        make_element(ElementType.TABLE, "| A |"),
    ]
    grouped = router._group_list_items(elements)
    assert len(grouped) == 3
    assert grouped[0].type == ElementType.TEXT
    assert grouped[1].type == ElementType.TEXT  # merged list
    assert grouped[2].type == ElementType.TABLE


def test_section_based_parents():
    """Each titled section becomes a parent."""
    chunker = ParentChildChunker()
    elements = [
        make_element(ElementType.TITLE, "# Section A"),
        make_element(ElementType.TEXT, "Content for section A. " * 60),
        make_element(ElementType.TITLE, "# Section B"),
        make_element(ElementType.TEXT, "Content for section B. " * 60),
    ]
    parents, children = chunker.chunk(elements, {})

    parent_sections = [p.section for p in parents]
    assert "Section A" in parent_sections
    assert "Section B" in parent_sections


def test_small_sections_merged():
    """Sections < parent_min_words are merged with the next section."""
    chunker = ParentChildChunker(parent_min_words=200)
    elements = [
        make_element(ElementType.TITLE, "# Tiny Section"),
        make_element(ElementType.TEXT, "Short content. " * 5),  # ~10 words
        make_element(ElementType.TITLE, "# Next Section"),
        make_element(ElementType.TEXT, "More content here. " * 50),  # ~150 words
    ]
    parents, children = chunker.chunk(elements, {})

    # Both sections should be merged into one parent since combined < 200 words
    text_parents = [p for p in parents if p.type == "text"]
    assert len(text_parents) == 1
    assert "Short content" in text_parents[0].content
    assert "More content here" in text_parents[0].content


def test_large_section_split_at_paragraphs():
    """Sections > parent_max_words are split at paragraph boundaries."""
    chunker = ParentChildChunker(parent_max_words=100)
    # Create a section with multiple paragraphs totaling > 100 words
    para1 = "First paragraph sentence. " * 15  # ~45 words
    para2 = "Second paragraph content. " * 15   # ~45 words
    para3 = "Third paragraph material. " * 15    # ~45 words
    big_text = f"{para1}\n\n{para2}\n\n{para3}"
    elements = [
        make_element(ElementType.TITLE, "# Big Section"),
        make_element(ElementType.TEXT, big_text),
    ]
    parents, children = chunker.chunk(elements, {})

    text_parents = [p for p in parents if p.type == "text"]
    assert len(text_parents) >= 2  # should be split into at least 2 parents


def test_no_title_document_still_produces_parents():
    """Documents with no titles fall back to paragraph-based parents."""
    chunker = ParentChildChunker(parent_max_words=50)
    para1 = "First paragraph. " * 20     # ~40 words
    para2 = "Second paragraph. " * 20    # ~40 words
    elements = [
        make_element(ElementType.TEXT, f"{para1}\n\n{para2}"),
    ]
    parents, children = chunker.chunk(elements, {})

    text_parents = [p for p in parents if p.type == "text"]
    assert len(text_parents) >= 2


def test_parent_word_count_within_range():
    """Parent word counts should respect min/max bounds (except last parent)."""
    chunker = ParentChildChunker(parent_min_words=50, parent_max_words=200)
    sections = []
    for i in range(5):
        sections.append(make_element(ElementType.TITLE, f"# Section {i}"))
        sections.append(make_element(ElementType.TEXT, f"Content for section {i}. " * 25))
    parents, _ = chunker.chunk(sections, {})

    text_parents = [p for p in parents if p.type == "text"]
    # All but the last parent should be >= min
    for p in text_parents[:-1]:
        assert p.word_count >= 50
