import pytest
from unittest.mock import AsyncMock

from backend.app.pipeline.multimodal.table_representer import (
    TableRepresenter,
    TableRepresentations,
)


async def test_represent_from_html():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = (
        "The table shows Q1 revenue of 12.4M and Q2 revenue of 15.1M."
    )
    representer = TableRepresenter(llm=mock_llm)
    result = await representer.represent(
        table_html="<table><tr><th>Quarter</th><th>Revenue</th></tr>"
        "<tr><td>Q1</td><td>12.4M</td></tr>"
        "<tr><td>Q2</td><td>15.1M</td></tr></table>",
        section_context="Financial Performance",
    )
    assert isinstance(result, TableRepresentations)
    assert "12.4M" in result.markdown
    assert "<table>" in result.html
    assert "revenue" in result.natural_language.lower()


async def test_represent_generates_markdown_from_html():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "A simple table."
    representer = TableRepresenter(llm=mock_llm)
    result = await representer.represent(
        table_html="<table><tr><th>A</th><th>B</th></tr>"
        "<tr><td>1</td><td>2</td></tr></table>",
    )
    assert "| A" in result.markdown
    assert "| 1" in result.markdown


async def test_represent_empty_html_raises():
    mock_llm = AsyncMock()
    representer = TableRepresenter(llm=mock_llm)
    with pytest.raises(ValueError, match="empty"):
        await representer.represent(table_html="")


def test_html_to_markdown_basic():
    representer = TableRepresenter.__new__(TableRepresenter)
    md = representer._html_to_markdown(
        "<table><tr><th>Name</th><th>Age</th></tr>"
        "<tr><td>Alice</td><td>30</td></tr></table>"
    )
    assert "| Name" in md
    assert "| Alice" in md
    assert "---" in md
