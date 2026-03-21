from typing import Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Structured output for document-level metadata extraction."""

    title: str = Field(description="Document title or filename if unclear")
    doc_type: str = Field(
        description="One of: policy|contract|report|manual|invoice|form|other"
    )
    language: str = Field(description="ISO 639-1 code e.g. en, vi, de")
    date: Optional[str] = Field(default=None, description="YYYY-MM-DD or null")
    organization: Optional[str] = Field(
        default=None, description="Company/org name or null"
    )
    summary: str = Field(description="One sentence summary")
