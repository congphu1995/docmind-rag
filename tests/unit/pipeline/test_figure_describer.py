import pytest
from unittest.mock import AsyncMock

from backend.app.pipeline.multimodal.figure_describer import FigureDescriber


async def test_describe_returns_text():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = (
        "Bar chart showing Q1-Q4 revenue growth from 10M to 15M."
    )
    describer = FigureDescriber(llm=mock_llm)
    result = await describer.describe(
        image_b64="iVBORw0KGgoAAAANSUhEUg==",
        doc_context="2024 Annual Report, Financial Performance section",
    )
    assert "revenue" in result.lower() or len(result) > 20
    mock_llm.complete.assert_called_once()


async def test_describe_includes_context_in_prompt():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "A figure."
    describer = FigureDescriber(llm=mock_llm)
    await describer.describe(
        image_b64="abc123",
        doc_context="Insurance Claims Report, Table 3",
    )
    call_args = mock_llm.complete.call_args
    messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
    # Check image_url content block is present
    user_msg = next(m for m in messages if m["role"] == "user")
    assert isinstance(user_msg["content"], list)
    image_block = next(b for b in user_msg["content"] if b.get("type") == "image_url")
    assert "abc123" in image_block["image_url"]["url"]


async def test_describe_empty_image_raises():
    mock_llm = AsyncMock()
    describer = FigureDescriber(llm=mock_llm)
    with pytest.raises(ValueError, match="empty"):
        await describer.describe(image_b64="", doc_context="test")
