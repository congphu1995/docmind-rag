"""
Extract document-level metadata with one LLM call.
Result stored with every chunk in Qdrant payload.
Uses structured output — returns a validated Pydantic model.
"""

from backend.app.pipeline.base.llm_client import BaseLLMClient
from backend.app.pipeline.base.parser import ElementType, ParsedElement
from backend.app.pipeline.prompts import METADATA_EXTRACTION_PROMPT
from backend.app.schemas.pipeline import DocumentMetadata


class MetadataExtractor:
    def __init__(self, llm: BaseLLMClient):
        self._llm = llm

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
            result = await self._llm.complete_structured(
                messages=[
                    {
                        "role": "user",
                        "content": METADATA_EXTRACTION_PROMPT.format(
                            sample_text=sample_text
                        ),
                    }
                ],
                response_model=DocumentMetadata,
                max_tokens=200,
                temperature=0,
            )
            return result.model_dump()
        except Exception:
            return DocumentMetadata(
                title="Unknown",
                doc_type="document",
                language="en",
                summary="Metadata extraction failed",
            ).model_dump()
