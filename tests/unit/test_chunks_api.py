from unittest.mock import MagicMock

from backend.app.api.chunks import _build_chunk_tree


def test_build_chunk_tree_groups_children():
    parents = [
        MagicMock(
            chunk_id="p1",
            content_raw="parent content",
            content_markdown="# parent",
            content_html="<p>parent</p>",
            type="text",
            page=1,
            section="Introduction",
            language="en",
            word_count=100,
        )
    ]

    children = [
        {
            "chunk_id": "c1",
            "parent_id": "p1",
            "content": "child enriched",
            "content_raw": "child raw",
            "type": "text",
            "page": 1,
            "section": "Introduction",
        },
        {
            "chunk_id": "c2",
            "parent_id": "p1",
            "content": "child 2 enriched",
            "content_raw": "child 2 raw",
            "type": "text",
            "page": 1,
            "section": "Introduction",
        },
    ]

    tree = _build_chunk_tree(parents, children)
    assert len(tree) == 1
    assert tree[0]["chunk_id"] == "p1"
    assert len(tree[0]["children"]) == 2


def test_build_chunk_tree_handles_orphans():
    parents = []
    children = [
        {
            "chunk_id": "atomic1",
            "parent_id": "",
            "content_raw": "table data here",
            "type": "table",
            "page": 3,
            "section": "Results",
        }
    ]

    tree = _build_chunk_tree(parents, children)
    assert len(tree) == 1
    assert tree[0]["type"] == "table"
    assert tree[0]["children"] == []
