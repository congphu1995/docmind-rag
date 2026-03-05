"""
Extract document-level metadata with one LLM call.
Result stored with every chunk in Qdrant payload.
Uses OpenAI structured output — returns a validated Pydantic model, no manual JSON parsing.
"""
from openai import AsyncOpenAI

from backend.app.core.config import settings
from backend.app.pipeline.base.parser import ElementType, ParsedElement
from backend.app.pipeline.prompts import METADATA_EXTRACTION_PROMPT
from backend.app.schemas.pipeline import DocumentMetadata


class MetadataExtractor:
    # TODO(week2): Replace direct AsyncOpenAI with BaseLLMClient injection

    def __init__(self):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def extract(self, elements: list[ParsedElement]) -> dict:
        """Extract doc-level metadata from first ~2000 chars of document."""
        sample_text = " ".join(
            el.content
            for el in elements[:30]
            if el.type in (ElementType.TEXT, ElementType.TITLE)
        )[:2000]

        if not sample_text.strip():
            return DocumentMetadata(
                title="Unknown",
                doc_type="document",
                language="en",
                summary="No content extracted",
            ).model_dump()

        try:
            response = await self._client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                max_tokens=200,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": METADATA_EXTRACTION_PROMPT.format(
                            sample_text=sample_text
                        ),
                    }
                ],
                response_format=DocumentMetadata,
            )
            return response.choices[0].message.parsed.model_dump()
        except Exception:
            return DocumentMetadata(
                title="Unknown",
                doc_type="document",
                language="en",
                summary="Metadata extraction failed",
            ).model_dump()
