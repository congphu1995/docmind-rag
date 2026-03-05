# Week 1 — Ingestion Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upload a PDF → chunks stored in Qdrant + parent chunks in PostgreSQL, with correct metadata and enrichment applied. `POST /api/v1/documents/upload` works end-to-end.

**Architecture:** Strategy pattern — every component (parser, chunker, embedder) sits behind an ABC. ParserFactory auto-selects Docling or PyMuPDF based on file analysis. Parent-child chunking stores large parents in PostgreSQL and small embedded children in Qdrant. Celery handles async ingestion.

**Tech Stack:** FastAPI, SQLAlchemy (async), Qdrant, PostgreSQL, Celery/Redis, Docling, PyMuPDF, OpenAI embeddings, structlog

> **Week 2 Refactor Note:** `ContextEnricher` and `MetadataExtractor` call `AsyncOpenAI` directly in Week 1 because `BaseLLMClient` concrete implementations (`ClaudeClient`, `OpenAIClient`, `LLMFactory`) are built in Week 2 Day 1. Once those are available, refactor both classes to accept a `BaseLLMClient` via constructor injection instead of instantiating `AsyncOpenAI` internally.

---

## Task 1: Project Scaffold — Git, UV, Directory Structure

**Files:**
- Create: `backend/pyproject.toml` (via uv init)
- Create: All directory structure + `__init__.py` files
- Create: `.env.example`
- Create: `docker-compose.yml`
- Create: `.gitignore`

**Step 1: Initialize git repo**

```bash
cd /home/ncphu/learning/docmind-rag
git init
```

**Step 2: Initialize Python project with uv**

```bash
cd /home/ncphu/learning/docmind-rag
uv init --python 3.11
```

**Step 3: Add dependencies**

```bash
uv add fastapi "uvicorn[standard]" pydantic pydantic-settings \
       "sqlalchemy[asyncio]" asyncpg alembic \
       qdrant-client openai \
       "celery[redis]" structlog python-multipart \
       docling pymupdf4llm pymupdf
```

```bash
uv add --dev pytest pytest-asyncio httpx pytest-mock ruff mypy
```

**Step 4: Create full directory structure**

```bash
mkdir -p backend/app/{api,core,models,schemas,services}
mkdir -p backend/app/pipeline/{base,parsers,chunkers,embedders,llm,rerankers,multimodal}
mkdir -p backend/app/{agent/nodes,vectorstore,workers}
mkdir -p frontend/src/{pages,components/{chat,documents},stores,hooks,api}
mkdir -p eval/{datasets,results,notebooks}
mkdir -p scripts tests/{unit/{pipeline,agent},integration,fixtures}
```

Create all `__init__.py`:

```bash
find backend/app -type d -exec touch {}/__init__.py \;
touch tests/__init__.py tests/unit/__init__.py tests/unit/pipeline/__init__.py
touch tests/unit/agent/__init__.py tests/integration/__init__.py
```

**Step 5: Create `.gitignore`**

```
__pycache__/
*.pyc
.env
*.egg-info/
dist/
.venv/
.mypy_cache/
.pytest_cache/
.ruff_cache/
*.pdf
!tests/fixtures/*.pdf
```

**Step 6: Create `.env.example`**

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
QDRANT_HOST=localhost
QDRANT_PORT=6333
POSTGRES_URL=postgresql+asyncpg://docmind:docmind@localhost:5432/docmind
REDIS_URL=redis://localhost:6379/0
DEBUG=false
TZ=Asia/Ho_Chi_Minh
```

**Step 7: Create `docker-compose.yml`**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - TZ=Asia/Ho_Chi_Minh

  postgres:
    image: ai-docker-registry.dai-ichi-life.com.vn:5000/postgres:15-alpine
    environment:
      POSTGRES_USER: docmind
      POSTGRES_PASSWORD: docmind
      POSTGRES_DB: docmind
      TZ: Asia/Ho_Chi_Minh
      PGTZ: Asia/Ho_Chi_Minh
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:latest
    ports:
      - "6379:6379"
    environment:
      - TZ=Asia/Ho_Chi_Minh

volumes:
  qdrant_data:
  postgres_data:
```

**Step 8: Create `pytest.ini` (or `pyproject.toml` section)**

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
]
testpaths = ["tests"]
```

**Step 9: Verify**

```bash
uv sync
uv run python -c "import fastapi; print('OK')"
```

**Step 10: Commit**

```bash
git add -A
git commit -m "feat: project scaffold with uv, directories, docker-compose"
```

---

## Task 2: Core Config + Exceptions + Logging

**Files:**
- Create: `backend/app/core/config.py`
- Create: `backend/app/core/exceptions.py`
- Create: `backend/app/core/logging.py`

**Step 1: Create `backend/app/core/config.py`**

```python
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # App
    app_name: str = "DocMind RAG"
    debug: bool = False

    # LLM
    openai_api_key: str
    anthropic_api_key: str = ""
    default_llm: Literal["claude", "openai"] = "openai"

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Parser
    parser_strategy: Literal["docling", "pymupdf", "auto"] = "auto"

    # Chunker (word counts)
    parent_chunk_size: int = 800
    child_chunk_size: int = 150
    chunk_overlap: int = 15

    # Retrieval
    retrieval_top_k: int = 20
    retrieval_top_n: int = 5
    retrieval_score_threshold: float = 0.4
    retrieval_quality_threshold: float = 0.6

    # HyDE
    hyde_model: str = "gpt-4o-mini"
    hyde_max_tokens: int = 150

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "docmind_chunks"

    # PostgreSQL
    postgres_url: str = "postgresql+asyncpg://docmind:docmind@localhost:5432/docmind"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "docmind-files"

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3001"

    model_config = {"env_file": ".env"}


settings = Settings()
```

**Step 2: Create `backend/app/core/exceptions.py`**

```python
class DocMindError(Exception):
    """Base exception for all DocMind errors."""
    pass


class ParserError(DocMindError):
    pass


class EncryptedDocumentError(ParserError):
    """Raised when a PDF is password-protected."""
    pass


class UnsupportedFileTypeError(ParserError):
    pass


class ChunkingError(DocMindError):
    pass


class EmbeddingError(DocMindError):
    pass


class VectorStoreError(DocMindError):
    pass


class IngestionError(DocMindError):
    pass


class RetrievalError(DocMindError):
    pass
```

**Step 3: Create `backend/app/core/logging.py`**

```python
import logging

import structlog

from backend.app.core.config import settings


def configure_logging():
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG if settings.debug else logging.INFO
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


logger = structlog.get_logger()
```

**Step 4: Verify config validates on missing key**

```bash
unset OPENAI_API_KEY && uv run python -c "from backend.app.core.config import Settings; Settings()" 2>&1 | grep -i "validation"
```

Expected: ValidationError about missing openai_api_key.

**Step 5: Commit**

```bash
git add backend/app/core/
git commit -m "feat: core config, exceptions, structlog logging"
```

---

## Task 3: Core Schemas — ParsedElement + Chunk + ABCs

**Files:**
- Create: `backend/app/pipeline/base/parser.py`
- Create: `backend/app/pipeline/base/chunker.py`
- Create: `backend/app/pipeline/base/embedder.py`
- Create: `backend/app/pipeline/base/reranker.py`
- Create: `backend/app/pipeline/base/retriever.py`

**Step 1: Create `backend/app/pipeline/base/parser.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    TITLE = "title"
    LIST_ITEM = "list_item"
    CODE = "code"
    SCANNED = "scanned"
    CAPTION = "caption"


@dataclass
class ParsedElement:
    type: ElementType
    content: str
    page: int
    doc_id: str
    doc_name: str

    section_title: Optional[str] = None
    level: Optional[int] = None
    reading_order: Optional[int] = None
    bbox: Optional[tuple] = None

    image_b64: Optional[str] = None
    table_html: Optional[str] = None
    table_df: Optional[pd.DataFrame] = None

    confidence: Optional[float] = None
    language: Optional[str] = None
    is_scanned: bool = False

    parser_used: str = ""

    def is_atomic(self) -> bool:
        """Tables, figures, code are never split."""
        return self.type in (ElementType.TABLE, ElementType.FIGURE, ElementType.CODE)

    def is_structural_boundary(self) -> bool:
        """Titles mark section boundaries, not chunks themselves."""
        return self.type == ElementType.TITLE

    def word_count(self) -> int:
        return len(self.content.split())


@dataclass
class ParserCapabilities:
    handles_scanned: bool = False
    handles_tables: bool = True
    handles_figures: bool = False
    max_pages: Optional[int] = None
    output_confidence: bool = False


class BaseParser(ABC):
    """
    Abstract base — all parsers implement this interface.
    Services import BaseParser only, never a concrete class.
    """

    @abstractmethod
    async def parse(
        self,
        file_path: str,
        doc_id: str,
        doc_name: str,
        **kwargs,
    ) -> list[ParsedElement]:
        ...

    @abstractmethod
    def supports(self, file_ext: str) -> bool:
        ...

    @abstractmethod
    def get_capabilities(self) -> ParserCapabilities:
        ...
```

**Step 2: Create `backend/app/pipeline/base/chunker.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import uuid

from backend.app.pipeline.base.parser import ParsedElement


@dataclass
class Chunk:
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    doc_id: str = ""
    doc_name: str = ""

    content: str = ""
    content_raw: str = ""
    content_markdown: Optional[str] = None
    content_html: Optional[str] = None

    type: str = "text"
    page: int = 0
    section: str = ""
    language: str = "en"
    word_count: int = 0
    is_parent: bool = False

    metadata: dict = field(default_factory=dict)

    def qdrant_payload(self) -> dict:
        """Flat dict for Qdrant payload storage."""
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "content_raw": self.content_raw,
            "content_markdown": self.content_markdown,
            "type": self.type,
            "page": self.page,
            "section": self.section,
            "language": self.language,
            "word_count": self.word_count,
            **self.metadata,
        }


class BaseChunker(ABC):
    @abstractmethod
    def chunk(
        self,
        elements: list[ParsedElement],
        doc_metadata: dict,
    ) -> tuple[list[Chunk], list[Chunk]]:
        """
        Returns (parent_chunks, child_chunks).
        parent_chunks → PostgreSQL
        child_chunks  → Qdrant (embedded)
        """
        ...
```

**Step 3: Create `backend/app/pipeline/base/embedder.py`**

```python
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    async def embed_single(self, text: str) -> list[float]:
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        ...
```

**Step 4: Create `backend/app/pipeline/base/reranker.py`**

```python
from abc import ABC, abstractmethod


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_n: int,
    ) -> list[dict]:
        ...
```

**Step 5: Create `backend/app/pipeline/base/retriever.py`**

```python
from abc import ABC, abstractmethod
from typing import Optional


class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query_vector: list[float],
        top_k: int,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        ...
```

**Step 6: Verify imports work**

```bash
uv run python -c "
from backend.app.pipeline.base.parser import BaseParser, ParsedElement, ElementType, ParserCapabilities
from backend.app.pipeline.base.chunker import BaseChunker, Chunk
from backend.app.pipeline.base.embedder import BaseEmbedder
from backend.app.pipeline.base.reranker import BaseReranker
from backend.app.pipeline.base.retriever import BaseRetriever
print('All ABCs importable')
"
```

**Step 7: Commit**

```bash
git add backend/app/pipeline/base/
git commit -m "feat: core schemas (ParsedElement, Chunk) and all ABCs"
```

---

## Task 4: Schema Unit Tests

**Files:**
- Create: `tests/unit/pipeline/test_schemas.py`

**Step 1: Write tests**

```python
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
```

**Step 2: Run tests**

```bash
uv run pytest tests/unit/pipeline/test_schemas.py -v
```

Expected: All pass.

**Step 3: Commit**

```bash
git add tests/
git commit -m "test: unit tests for ParsedElement and Chunk schemas"
```

---

## Task 5: PDFPreprocessor

**Files:**
- Create: `backend/app/pipeline/parsers/preprocessor.py`
- Create: `tests/unit/pipeline/test_preprocessor.py`

**Step 1: Write test**

```python
import pytest
from unittest.mock import patch, MagicMock

from backend.app.pipeline.parsers.preprocessor import PDFPreprocessor
from backend.app.core.exceptions import EncryptedDocumentError


def _mock_fitz_doc(
    page_count=5,
    text_len=500,
    rotation=0,
    is_encrypted=False,
    is_repaired=False,
    width=612,
    height=792,
):
    """Create a mock fitz document."""
    mock_page = MagicMock()
    mock_page.get_text.return_value = "x" * (text_len // max(page_count, 1))
    mock_page.rotation = rotation
    mock_page.rect.width = width
    mock_page.rect.height = height

    mock_doc = MagicMock()
    mock_doc.page_count = page_count
    mock_doc.is_encrypted = is_encrypted
    mock_doc.is_repaired = is_repaired
    mock_doc.__iter__ = lambda self: iter([mock_page] * page_count)
    mock_doc.__getitem__ = lambda self, i: mock_page
    return mock_doc


@patch("backend.app.pipeline.parsers.preprocessor.fitz.open")
@patch("backend.app.pipeline.parsers.preprocessor.Path")
def test_diagnose_normal_pdf(mock_path, mock_fitz_open):
    mock_fitz_open.return_value = _mock_fitz_doc()
    mock_path.return_value.stat.return_value.st_size = 1024 * 1024

    preprocessor = PDFPreprocessor()
    result = preprocessor.diagnose("test.pdf")

    assert result["is_scanned"] is False
    assert result["is_encrypted"] is False
    assert result["is_rotated"] is False
    assert result["page_count"] == 5


@patch("backend.app.pipeline.parsers.preprocessor.fitz.open")
@patch("backend.app.pipeline.parsers.preprocessor.Path")
def test_diagnose_scanned_pdf(mock_path, mock_fitz_open):
    mock_fitz_open.return_value = _mock_fitz_doc(text_len=10)
    mock_path.return_value.stat.return_value.st_size = 1024 * 1024

    preprocessor = PDFPreprocessor()
    result = preprocessor.diagnose("scanned.pdf")

    assert result["is_scanned"] is True


@patch("backend.app.pipeline.parsers.preprocessor.fitz.open")
@patch("backend.app.pipeline.parsers.preprocessor.Path")
def test_diagnose_rotated_pdf(mock_path, mock_fitz_open):
    mock_fitz_open.return_value = _mock_fitz_doc(rotation=90)
    mock_path.return_value.stat.return_value.st_size = 1024 * 1024

    preprocessor = PDFPreprocessor()
    result = preprocessor.diagnose("rotated.pdf")

    assert result["is_rotated"] is True


@patch("backend.app.pipeline.parsers.preprocessor.fitz.open")
@patch("backend.app.pipeline.parsers.preprocessor.Path")
def test_preprocess_encrypted_raises(mock_path, mock_fitz_open):
    mock_fitz_open.return_value = _mock_fitz_doc(is_encrypted=True)
    mock_path.return_value.stat.return_value.st_size = 1024 * 1024

    preprocessor = PDFPreprocessor()
    with pytest.raises(EncryptedDocumentError):
        preprocessor.preprocess("encrypted.pdf")
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/pipeline/test_preprocessor.py -v
```

Expected: FAIL (module not found).

**Step 3: Implement `backend/app/pipeline/parsers/preprocessor.py`**

```python
from pathlib import Path

import fitz

from backend.app.core.exceptions import EncryptedDocumentError
from backend.app.core.logging import logger


class PDFPreprocessor:

    def diagnose(self, file_path: str) -> dict:
        """Inspect PDF before parsing. Returns diagnosis dict."""
        doc = fitz.open(file_path)
        page = doc[0] if doc.page_count > 0 else None

        text_len = sum(len(p.get_text()) for p in doc) if doc.page_count > 0 else 0
        page_area = (page.rect.width * page.rect.height) if page else 1

        return {
            "is_scanned": text_len < 50 and doc.page_count > 0,
            "is_rotated": page.rotation != 0 if page else False,
            "is_encrypted": doc.is_encrypted,
            "is_corrupt": doc.is_repaired,
            "page_count": doc.page_count,
            "text_density": text_len / (page_area * doc.page_count + 1),
            "file_size_mb": Path(file_path).stat().st_size / (1024 * 1024),
        }

    def preprocess(self, file_path: str) -> str:
        """
        Run diagnosis and fix issues.
        Returns path to cleaned file (may be same as input).
        """
        diagnosis = self.diagnose(file_path)
        logger.info("pdf_diagnosis", file=file_path, **diagnosis)

        if diagnosis["is_encrypted"]:
            raise EncryptedDocumentError(
                f"Document is password-protected: {file_path}. "
                "Please provide an unencrypted version."
            )

        if diagnosis["is_rotated"]:
            file_path = self._deskew(file_path)
            logger.info("pdf_deskewed", file=file_path)

        return file_path

    def _deskew(self, file_path: str) -> str:
        """Normalize page rotation."""
        doc = fitz.open(file_path)
        out_path = file_path.replace(".pdf", "_deskewed.pdf")
        for page in doc:
            if page.rotation != 0:
                page.set_rotation(0)
        doc.save(out_path)
        return out_path
```

**Step 4: Run tests**

```bash
uv run pytest tests/unit/pipeline/test_preprocessor.py -v
```

Expected: All pass.

**Step 5: Commit**

```bash
git add backend/app/pipeline/parsers/preprocessor.py tests/unit/pipeline/test_preprocessor.py
git commit -m "feat: PDFPreprocessor with diagnosis and deskew"
```

---

## Task 6: ElementNormalizer

**Files:**
- Create: `backend/app/pipeline/parsers/normalizer.py`
- Create: `tests/unit/pipeline/test_normalizer.py`

**Step 1: Write test**

```python
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
```

**Step 2: Implement `backend/app/pipeline/parsers/normalizer.py`**

```python
"""
Converts raw parser output (varies by parser) → unified ParsedElement list.
All downstream code only ever sees ParsedElement.
"""
from typing import Optional

from backend.app.pipeline.base.parser import ElementType, ParsedElement


class ElementNormalizer:

    def from_docling(
        self, docling_result, doc_id: str, doc_name: str
    ) -> list[ParsedElement]:
        """Convert Docling Document object → List[ParsedElement]."""
        elements = []
        current_section = ""

        for item in docling_result.document.body.children:
            el_type = self._map_docling_type(item)
            content = self._extract_docling_content(item)

            if not content or not content.strip():
                continue

            if el_type == ElementType.TITLE:
                current_section = content.strip()

            elements.append(
                ParsedElement(
                    type=el_type,
                    content=content,
                    page=self._get_docling_page(item),
                    doc_id=doc_id,
                    doc_name=doc_name,
                    section_title=(
                        current_section if el_type != ElementType.TITLE else None
                    ),
                    table_html=(
                        self._get_table_html(item)
                        if el_type == ElementType.TABLE
                        else None
                    ),
                    parser_used="docling",
                )
            )

        return self._assign_reading_order(elements)

    def from_pymupdf(
        self, pages_data: list[dict], doc_id: str, doc_name: str
    ) -> list[ParsedElement]:
        """Convert pymupdf4llm page dict list → List[ParsedElement]."""
        elements = []
        current_section = ""

        for page_data in pages_data:
            page_num = page_data.get("metadata", {}).get("page", 0)
            md_content = page_data.get("text", "")

            for block in self._parse_markdown_blocks(md_content):
                el_type = block["type"]
                content = block["content"]

                if not content.strip():
                    continue

                if el_type == ElementType.TITLE:
                    current_section = content.strip("# ").strip()

                elements.append(
                    ParsedElement(
                        type=el_type,
                        content=content,
                        page=page_num,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        section_title=current_section,
                        parser_used="pymupdf",
                    )
                )

        return self._assign_reading_order(elements)

    def _map_docling_type(self, item) -> ElementType:
        type_map = {
            "section_header": ElementType.TITLE,
            "text": ElementType.TEXT,
            "table": ElementType.TABLE,
            "figure": ElementType.FIGURE,
            "list_item": ElementType.LIST_ITEM,
            "code": ElementType.CODE,
        }
        item_type = getattr(item, "label", "text")
        return type_map.get(str(item_type).lower(), ElementType.TEXT)

    def _parse_markdown_blocks(self, md: str) -> list[dict]:
        """Parse markdown string into typed blocks."""
        blocks = []
        current_table: list[str] = []
        in_table = False
        lines = md.split("\n")

        for line in lines:
            if line.startswith("#"):
                if in_table and current_table:
                    blocks.append(
                        {"type": ElementType.TABLE, "content": "\n".join(current_table)}
                    )
                    current_table = []
                    in_table = False
                blocks.append({"type": ElementType.TITLE, "content": line})
            elif line.startswith("|"):
                in_table = True
                current_table.append(line)
            elif in_table and not line.startswith("|"):
                blocks.append(
                    {"type": ElementType.TABLE, "content": "\n".join(current_table)}
                )
                current_table = []
                in_table = False
                if line.strip():
                    blocks.append({"type": ElementType.TEXT, "content": line})
            elif line.strip():
                blocks.append({"type": ElementType.TEXT, "content": line})

        if current_table:
            blocks.append(
                {"type": ElementType.TABLE, "content": "\n".join(current_table)}
            )

        return blocks

    def _get_docling_page(self, item) -> int:
        try:
            return item.prov[0].page_no - 1
        except (AttributeError, IndexError):
            return 0

    def _get_table_html(self, item) -> Optional[str]:
        try:
            return item.export_to_html()
        except Exception:
            return None

    def _extract_docling_content(self, item) -> str:
        try:
            return item.export_to_markdown()
        except Exception:
            return str(getattr(item, "text", ""))

    def _assign_reading_order(
        self, elements: list[ParsedElement]
    ) -> list[ParsedElement]:
        for i, el in enumerate(elements):
            el.reading_order = i
        return elements
```

**Step 3: Run tests**

```bash
uv run pytest tests/unit/pipeline/test_normalizer.py -v
```

**Step 4: Commit**

```bash
git add backend/app/pipeline/parsers/normalizer.py tests/unit/pipeline/test_normalizer.py
git commit -m "feat: ElementNormalizer — converts parser output to ParsedElement"
```

---

## Task 7: DoclingParser + PyMuPDFParser

**Files:**
- Create: `backend/app/pipeline/parsers/docling_parser.py`
- Create: `backend/app/pipeline/parsers/pymupdf_parser.py`

**Step 1: Create `backend/app/pipeline/parsers/docling_parser.py`**

```python
import time

from docling.document_converter import DocumentConverter

from backend.app.core.exceptions import ParserError
from backend.app.core.logging import logger
from backend.app.pipeline.base.parser import (
    BaseParser,
    ParsedElement,
    ParserCapabilities,
)
from backend.app.pipeline.parsers.normalizer import ElementNormalizer


class DoclingParser(BaseParser):

    def __init__(
        self, enable_ocr: bool = True, enable_table_structure: bool = True
    ):
        self._converter = DocumentConverter()
        self._normalizer = ElementNormalizer()
        self._enable_ocr = enable_ocr

    def supports(self, file_ext: str) -> bool:
        return file_ext.lower() in {".pdf", ".docx", ".txt", ".md", ".html"}

    def get_capabilities(self) -> ParserCapabilities:
        return ParserCapabilities(
            handles_scanned=self._enable_ocr,
            handles_tables=True,
            handles_figures=True,
            output_confidence=False,
        )

    async def parse(
        self, file_path: str, doc_id: str, doc_name: str, **kwargs
    ) -> list[ParsedElement]:
        start = time.perf_counter()
        logger.info("docling_parse_start", doc_id=doc_id, file=file_path)

        try:
            result = self._converter.convert(file_path)
            elements = self._normalizer.from_docling(result, doc_id, doc_name)

            elapsed = time.perf_counter() - start
            logger.info(
                "docling_parse_done",
                doc_id=doc_id,
                elements=len(elements),
                elapsed_s=round(elapsed, 2),
            )
            return elements

        except Exception as e:
            logger.error("docling_parse_failed", doc_id=doc_id, error=str(e))
            raise ParserError(f"Docling failed on {file_path}: {e}") from e
```

**Step 2: Create `backend/app/pipeline/parsers/pymupdf_parser.py`**

```python
import time

import pymupdf4llm

from backend.app.core.exceptions import ParserError
from backend.app.core.logging import logger
from backend.app.pipeline.base.parser import (
    BaseParser,
    ParsedElement,
    ParserCapabilities,
)
from backend.app.pipeline.parsers.normalizer import ElementNormalizer


class PyMuPDFParser(BaseParser):

    def __init__(self):
        self._normalizer = ElementNormalizer()

    def supports(self, file_ext: str) -> bool:
        return file_ext.lower() == ".pdf"

    def get_capabilities(self) -> ParserCapabilities:
        return ParserCapabilities(
            handles_scanned=False,
            handles_tables=True,
            handles_figures=False,
            output_confidence=False,
        )

    async def parse(
        self, file_path: str, doc_id: str, doc_name: str, **kwargs
    ) -> list[ParsedElement]:
        start = time.perf_counter()
        logger.info("pymupdf_parse_start", doc_id=doc_id, file=file_path)

        try:
            pages_data = pymupdf4llm.to_markdown(file_path, page_chunks=True)
            elements = self._normalizer.from_pymupdf(pages_data, doc_id, doc_name)

            elapsed = time.perf_counter() - start
            logger.info(
                "pymupdf_parse_done",
                doc_id=doc_id,
                elements=len(elements),
                elapsed_s=round(elapsed, 2),
            )
            return elements

        except Exception as e:
            logger.error("pymupdf_parse_failed", doc_id=doc_id, error=str(e))
            raise ParserError(f"PyMuPDF failed on {file_path}: {e}") from e
```

**Step 3: Commit**

```bash
git add backend/app/pipeline/parsers/docling_parser.py backend/app/pipeline/parsers/pymupdf_parser.py
git commit -m "feat: DoclingParser and PyMuPDFParser implementations"
```

---

## Task 8: ParserFactory

**Files:**
- Create: `backend/app/pipeline/parsers/factory.py`
- Create: `tests/unit/pipeline/test_parsers.py`

**Step 1: Write tests**

```python
import pytest
from unittest.mock import patch, MagicMock

from backend.app.pipeline.parsers.factory import ParserFactory
from backend.app.pipeline.base.parser import ElementType


def test_factory_creates_docling():
    parser = ParserFactory.create("docling")
    from backend.app.pipeline.parsers.docling_parser import DoclingParser
    assert isinstance(parser, DoclingParser)


def test_factory_creates_pymupdf():
    parser = ParserFactory.create("pymupdf")
    from backend.app.pipeline.parsers.pymupdf_parser import PyMuPDFParser
    assert isinstance(parser, PyMuPDFParser)


def test_factory_raises_on_unknown():
    with pytest.raises(ValueError, match="Unknown parser strategy"):
        ParserFactory.create("nonexistent")


def test_pymupdf_supports_only_pdf():
    from backend.app.pipeline.parsers.pymupdf_parser import PyMuPDFParser
    p = PyMuPDFParser()
    assert p.supports(".pdf") is True
    assert p.supports(".docx") is False


def test_docling_supports_multiple_formats():
    from backend.app.pipeline.parsers.docling_parser import DoclingParser
    p = DoclingParser()
    for ext in [".pdf", ".docx", ".txt", ".md"]:
        assert p.supports(ext) is True


@patch("backend.app.pipeline.parsers.factory.PDFPreprocessor")
def test_auto_select_scanned_returns_docling(mock_preprocessor):
    mock_preprocessor.return_value.diagnose.return_value = {
        "is_scanned": True,
        "page_count": 5,
    }
    parser = ParserFactory.auto_select("scanned.pdf")
    from backend.app.pipeline.parsers.docling_parser import DoclingParser
    assert isinstance(parser, DoclingParser)


@patch("backend.app.pipeline.parsers.factory.PDFPreprocessor")
def test_auto_select_large_clean_returns_pymupdf(mock_preprocessor):
    mock_preprocessor.return_value.diagnose.return_value = {
        "is_scanned": False,
        "page_count": 100,
    }
    parser = ParserFactory.auto_select("large.pdf")
    from backend.app.pipeline.parsers.pymupdf_parser import PyMuPDFParser
    assert isinstance(parser, PyMuPDFParser)


def test_auto_select_docx_returns_docling():
    parser = ParserFactory.auto_select("document.docx")
    from backend.app.pipeline.parsers.docling_parser import DoclingParser
    assert isinstance(parser, DoclingParser)
```

**Step 2: Implement `backend/app/pipeline/parsers/factory.py`**

```python
from pathlib import Path

from backend.app.core.config import settings
from backend.app.pipeline.base.parser import BaseParser
from backend.app.pipeline.parsers.docling_parser import DoclingParser
from backend.app.pipeline.parsers.preprocessor import PDFPreprocessor
from backend.app.pipeline.parsers.pymupdf_parser import PyMuPDFParser


class ParserFactory:

    @staticmethod
    def create(strategy: str = None) -> BaseParser:
        """Create parser by explicit strategy name."""
        strategy = strategy or settings.parser_strategy
        parsers = {
            "docling": DoclingParser,
            "pymupdf": PyMuPDFParser,
        }
        if strategy not in parsers and strategy != "auto":
            raise ValueError(
                f"Unknown parser strategy: {strategy}. Choose: {list(parsers.keys())}"
            )
        if strategy == "auto":
            return DoclingParser()
        return parsers[strategy]()

    @staticmethod
    def auto_select(file_path: str) -> BaseParser:
        """
        Select best parser based on file type + content analysis.
        Clean PDF → PyMuPDF (fast)
        Scanned PDF → Docling (OCR)
        DOCX / TXT → Docling
        """
        ext = Path(file_path).suffix.lower()

        if ext != ".pdf":
            return DoclingParser()

        diagnosis = PDFPreprocessor().diagnose(file_path)

        if diagnosis["is_scanned"]:
            return DoclingParser(enable_ocr=True)

        if diagnosis["page_count"] > 50:
            return PyMuPDFParser()

        return DoclingParser(enable_ocr=False)
```

**Step 3: Run tests**

```bash
uv run pytest tests/unit/pipeline/test_parsers.py -v
```

**Step 4: Commit**

```bash
git add backend/app/pipeline/parsers/factory.py tests/unit/pipeline/test_parsers.py
git commit -m "feat: ParserFactory with auto_select based on file analysis"
```

---

## Task 9: ParentChildChunker

**Files:**
- Create: `backend/app/pipeline/chunkers/parent_child_chunker.py`

**Step 1: Implement**

```python
import uuid

from backend.app.core.config import settings
from backend.app.pipeline.base.chunker import BaseChunker, Chunk
from backend.app.pipeline.base.parser import ElementType, ParsedElement


class ParentChildChunker(BaseChunker):
    """
    Splits text elements into parent-child hierarchy.
    Parent (~800 words) → PostgreSQL, sent to LLM
    Child  (~150 words) → Qdrant, used for retrieval

    Tables, figures, code → always atomic (single chunk, no splitting)
    Titles → section boundary marker, not a chunk
    """

    def __init__(
        self,
        parent_size: int = None,
        child_size: int = None,
        overlap: int = None,
    ):
        self.parent_size = parent_size or settings.parent_chunk_size
        self.child_size = child_size or settings.child_chunk_size
        self.overlap = overlap or settings.chunk_overlap

    def chunk(
        self,
        elements: list[ParsedElement],
        doc_metadata: dict,
    ) -> tuple[list[Chunk], list[Chunk]]:
        parent_chunks: list[Chunk] = []
        child_chunks: list[Chunk] = []

        current_section = "Introduction"
        text_buffer: list[ParsedElement] = []

        for el in elements:
            if el.is_structural_boundary():
                if text_buffer:
                    parents, children = self._flush_text_buffer(
                        text_buffer, current_section, doc_metadata
                    )
                    parent_chunks.extend(parents)
                    child_chunks.extend(children)
                    text_buffer = []
                current_section = el.content.strip("# ").strip()

            elif el.is_atomic():
                if text_buffer:
                    parents, children = self._flush_text_buffer(
                        text_buffer, current_section, doc_metadata
                    )
                    parent_chunks.extend(parents)
                    child_chunks.extend(children)
                    text_buffer = []
                atomic = self._make_atomic_chunk(el, current_section, doc_metadata)
                parent_chunks.append(atomic)
                child_chunks.append(atomic)

            else:
                text_buffer.append(el)
                current_words = sum(
                    self._count_words(e.content) for e in text_buffer
                )
                if current_words >= self.parent_size * 1.2:
                    parents, children = self._flush_text_buffer(
                        text_buffer, current_section, doc_metadata
                    )
                    parent_chunks.extend(parents)
                    child_chunks.extend(children)
                    text_buffer = []

        if text_buffer:
            parents, children = self._flush_text_buffer(
                text_buffer, current_section, doc_metadata
            )
            parent_chunks.extend(parents)
            child_chunks.extend(children)

        return parent_chunks, child_chunks

    def _flush_text_buffer(
        self,
        buffer: list[ParsedElement],
        section: str,
        doc_metadata: dict,
    ) -> tuple[list[Chunk], list[Chunk]]:
        if not buffer:
            return [], []

        full_text = "\n\n".join(el.content for el in buffer)
        page = buffer[0].page
        doc_id = buffer[0].doc_id
        doc_name = buffer[0].doc_name
        language = buffer[0].language or "en"

        parent_id = str(uuid.uuid4())
        parent = Chunk(
            chunk_id=parent_id,
            parent_id=None,
            doc_id=doc_id,
            doc_name=doc_name,
            content=full_text,
            content_raw=full_text,
            type="text",
            page=page,
            section=section,
            language=language,
            word_count=self._count_words(full_text),
            is_parent=True,
            metadata=doc_metadata,
        )

        children = []
        child_texts = self._split_into_children(full_text)

        for i, child_text in enumerate(child_texts):
            child = Chunk(
                chunk_id=str(uuid.uuid4()),
                parent_id=parent_id,
                doc_id=doc_id,
                doc_name=doc_name,
                content=child_text,
                content_raw=child_text,
                type="text",
                page=page,
                section=section,
                language=language,
                word_count=self._count_words(child_text),
                is_parent=False,
                metadata={**doc_metadata, "chunk_index": i},
            )
            children.append(child)

        return [parent], children

    def _make_atomic_chunk(
        self,
        el: ParsedElement,
        section: str,
        doc_metadata: dict,
    ) -> Chunk:
        content = el.content
        if el.type == ElementType.TABLE and el.table_html:
            content_markdown = el.content
            content_html = el.table_html
        else:
            content_markdown = None
            content_html = None

        return Chunk(
            chunk_id=str(uuid.uuid4()),
            parent_id=None,
            doc_id=el.doc_id,
            doc_name=el.doc_name,
            content=content,
            content_raw=content,
            content_markdown=content_markdown,
            content_html=content_html,
            type=el.type.value,
            page=el.page,
            section=section,
            language=el.language or "en",
            word_count=self._count_words(content),
            is_parent=True,
            metadata=doc_metadata,
        )

    def _split_into_children(self, text: str) -> list[str]:
        """Split text into child-sized chunks by word count with overlap."""
        words = text.split()
        if len(words) <= self.child_size:
            return [text]

        children = []
        start = 0
        while start < len(words):
            end = min(start + self.child_size, len(words))
            child_text = " ".join(words[start:end])
            children.append(child_text)
            start += self.child_size - self.overlap

        return children

    @staticmethod
    def _count_words(text: str) -> int:
        return len(text.split())
```

**Step 2: Commit**

```bash
git add backend/app/pipeline/chunkers/parent_child_chunker.py
git commit -m "feat: ParentChildChunker with token-based splitting"
```

---

## Task 10: Pydantic Schemas + Prompts File

**Files:**
- Create: `backend/app/schemas/pipeline.py`
- Create: `backend/app/pipeline/prompts.py`

**Step 1: Create `backend/app/schemas/pipeline.py`**

Structured output models for LLM responses — no manual JSON parsing.

```python
from typing import Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Structured output for document-level metadata extraction."""
    title: str = Field(description="Document title or filename if unclear")
    doc_type: str = Field(description="One of: policy|contract|report|manual|invoice|form|other")
    language: str = Field(description="ISO 639-1 code e.g. en, vi, de")
    date: Optional[str] = Field(default=None, description="YYYY-MM-DD or null")
    organization: Optional[str] = Field(default=None, description="Company/org name or null")
    summary: str = Field(description="One sentence summary")
```

**Step 2: Create `backend/app/pipeline/prompts.py`**

All prompts in one file — versioned, not inline.

```python
"""
All prompts used in the pipeline.
Never put prompts inline in business logic.
"""

ENRICHMENT_PROMPT = """\
Given this document context, write 1-2 sentences that situate the chunk \
within the document. Be specific about the section and topic. \
Output ONLY the situating sentences — no preamble, no explanation.

Document title: {doc_title}
Document type: {doc_type}
Current section: {section}

Chunk text:
{chunk_text}"""

METADATA_EXTRACTION_PROMPT = """\
Extract metadata from this document excerpt.

Document:
{sample_text}"""
```

**Step 3: Commit**

```bash
git add backend/app/schemas/pipeline.py backend/app/pipeline/prompts.py
git commit -m "feat: Pydantic structured output schemas + centralized prompts"
```

---

## Task 11: SmartRouter + QualityFilter + ContextEnricher

**Files:**
- Create: `backend/app/pipeline/chunkers/smart_router.py`
- Create: `backend/app/pipeline/chunkers/quality_filter.py`
- Create: `backend/app/pipeline/chunkers/enricher.py`

**Step 1: Create `backend/app/pipeline/chunkers/smart_router.py`**

```python
"""
Routes each ParsedElement to the correct chunking strategy.
Entry point for all chunking — nothing calls ParentChildChunker directly.
"""
from backend.app.core.logging import logger
from backend.app.pipeline.base.chunker import Chunk
from backend.app.pipeline.base.parser import ElementType, ParsedElement
from backend.app.pipeline.chunkers.parent_child_chunker import ParentChildChunker


class SmartRouter:

    def __init__(self, chunker: ParentChildChunker = None):
        self._chunker = chunker or ParentChildChunker()

    def route(
        self,
        elements: list[ParsedElement],
        doc_metadata: dict,
    ) -> tuple[list[Chunk], list[Chunk]]:
        elements = self._group_list_items(elements)

        type_counts: dict[str, int] = {}
        for el in elements:
            type_counts[el.type.value] = type_counts.get(el.type.value, 0) + 1
        logger.info("chunking_routing", **type_counts)

        parents, children = self._chunker.chunk(elements, doc_metadata)

        logger.info(
            "chunking_done",
            doc_id=doc_metadata.get("doc_id", ""),
            parents=len(parents),
            children=len(children),
        )
        return parents, children

    def _group_list_items(
        self, elements: list[ParsedElement]
    ) -> list[ParsedElement]:
        """Merge consecutive list items into one TEXT element."""
        result = []
        list_buffer: list[ParsedElement] = []

        for el in elements:
            if el.type == ElementType.LIST_ITEM:
                list_buffer.append(el)
            else:
                if list_buffer:
                    merged = self._merge_list_items(list_buffer)
                    result.append(merged)
                    list_buffer = []
                result.append(el)

        if list_buffer:
            result.append(self._merge_list_items(list_buffer))

        return result

    def _merge_list_items(
        self, items: list[ParsedElement]
    ) -> ParsedElement:
        merged_content = "\n".join(f"• {el.content.strip()}" for el in items)
        first = items[0]
        return ParsedElement(
            type=ElementType.TEXT,
            content=merged_content,
            page=first.page,
            doc_id=first.doc_id,
            doc_name=first.doc_name,
            section_title=first.section_title,
            language=first.language,
            parser_used=first.parser_used,
        )
```

**Step 2: Create `backend/app/pipeline/chunkers/quality_filter.py`**

```python
import re

from backend.app.core.logging import logger
from backend.app.pipeline.base.chunker import Chunk


class QualityFilter:

    MIN_WORDS = 15
    MIN_ALPHA_RATIO = 0.4
    MAX_NUMERIC_LINE_RATIO = 0.8

    def filter(self, chunks: list[Chunk]) -> list[Chunk]:
        """Remove noise chunks before indexing."""
        before = len(chunks)
        filtered = [c for c in chunks if self._is_quality(c)]
        removed = before - len(filtered)

        if removed > 0:
            logger.info("quality_filter", removed=removed, kept=len(filtered))

        return filtered

    def _is_quality(self, chunk: Chunk) -> bool:
        if chunk.is_parent or chunk.type != "text":
            return True

        text = chunk.content_raw
        words = text.split()

        if len(words) < self.MIN_WORDS:
            return False

        alpha_chars = sum(c.isalpha() for c in text)
        if alpha_chars / max(len(text), 1) < self.MIN_ALPHA_RATIO:
            return False

        if re.search(r"(.)\1{4,}", text):
            return False

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        numeric_lines = sum(
            1 for line in lines if re.match(r"^[\d\s\.,\-\+%$€£]+$", line)
        )
        if lines and numeric_lines / len(lines) > self.MAX_NUMERIC_LINE_RATIO:
            return False

        return True
```

**Step 3: Create `backend/app/pipeline/chunkers/enricher.py`**

```python
"""
Prepend situating context to each child chunk before embedding.
Anthropic technique — +15-20% retrieval precision at one-time index cost.
"""
import asyncio

from openai import AsyncOpenAI

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.pipeline.base.chunker import Chunk
from backend.app.pipeline.prompts import ENRICHMENT_PROMPT


class ContextEnricher:
    # TODO(week2): Replace direct AsyncOpenAI with BaseLLMClient injection

    def __init__(self):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = "gpt-4o-mini"

    async def enrich_batch(
        self,
        chunks: list[Chunk],
        doc_metadata: dict,
        concurrency: int = 10,
    ) -> list[Chunk]:
        """
        Enrich all child chunks in a document.
        Skips parents and non-text chunks.
        """
        to_enrich = [c for c in chunks if not c.is_parent and c.type == "text"]
        skip = [c for c in chunks if c.is_parent or c.type != "text"]

        if not to_enrich:
            return chunks

        logger.info(
            "enrichment_start",
            chunks=len(to_enrich),
            doc_id=doc_metadata.get("doc_id"),
        )

        semaphore = asyncio.Semaphore(concurrency)
        tasks = [
            self._enrich_one(chunk, doc_metadata, semaphore) for chunk in to_enrich
        ]
        enriched = await asyncio.gather(*tasks, return_exceptions=True)

        final_enriched = []
        for chunk, result in zip(to_enrich, enriched):
            if isinstance(result, Exception):
                logger.warning(
                    "enrichment_failed", chunk_id=chunk.chunk_id, error=str(result)
                )
                final_enriched.append(chunk)
            else:
                final_enriched.append(result)

        logger.info("enrichment_done", doc_id=doc_metadata.get("doc_id"))
        return skip + final_enriched

    async def _enrich_one(
        self,
        chunk: Chunk,
        doc_metadata: dict,
        semaphore: asyncio.Semaphore,
    ) -> Chunk:
        async with semaphore:
            prompt = ENRICHMENT_PROMPT.format(
                doc_title=doc_metadata.get("title", chunk.doc_name),
                doc_type=doc_metadata.get("doc_type", "document"),
                section=chunk.section or "Unknown",
                chunk_text=chunk.content_raw[:500],
            )

            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=80,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )

            context_sentence = response.choices[0].message.content.strip()
            enriched_content = f"{context_sentence}\n\n{chunk.content_raw}"

            return Chunk(
                **{k: v for k, v in chunk.__dict__.items() if k != "content"},
                content=enriched_content,
            )
```

**Step 4: Commit**

```bash
git add backend/app/pipeline/chunkers/
git commit -m "feat: SmartRouter, QualityFilter, ContextEnricher"
```

---

## Task 11: Chunker Unit Tests

**Files:**
- Create: `tests/unit/pipeline/test_chunkers.py`

**Step 1: Write tests**

```python
import pytest

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
    chunker = ParentChildChunker(parent_size=50, child_size=15)
    long_text = "word " * 200
    elements = [make_element(ElementType.TEXT, long_text)]
    parents, children = chunker.chunk(elements, {})

    parent_ids = {p.chunk_id for p in parents if p.is_parent}
    for child in children:
        if not child.is_parent:
            assert child.parent_id in parent_ids


def test_short_text_produces_single_child():
    chunker = ParentChildChunker(parent_size=800, child_size=150)
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
```

**Step 2: Run tests**

```bash
uv run pytest tests/unit/pipeline/test_chunkers.py -v
```

**Step 3: Commit**

```bash
git add tests/unit/pipeline/test_chunkers.py
git commit -m "test: chunker unit tests — atomic tables, parent-child, quality filter"
```

---

## Task 12: Database Models + Engine

**Files:**
- Create: `backend/app/models/document.py`
- Create: `backend/app/core/database.py`

**Step 1: Create `backend/app/models/document.py`**

```python
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(String, primary_key=True)
    doc_name = Column(String, nullable=False)
    file_path = Column(String)
    language = Column(String, default="en")
    doc_type = Column(String, default="document")
    page_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    parser_used = Column(String)
    status = Column(String, default="processing")
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ParentChunk(Base):
    __tablename__ = "parent_chunks"

    chunk_id = Column(String, primary_key=True)
    doc_id = Column(String, nullable=False, index=True)
    content_raw = Column(Text, nullable=False)
    content_markdown = Column(Text)
    content_html = Column(Text)
    type = Column(String, default="text")
    page = Column(Integer, default=0)
    section = Column(String, default="")
    language = Column(String, default="en")
    word_count = Column(Integer, default=0)
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
```

**Step 2: Create `backend/app/core/database.py`**

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings
from backend.app.models.document import Base

engine = create_async_engine(
    settings.postgres_url,
    echo=settings.debug,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

**Step 3: Commit**

```bash
git add backend/app/models/document.py backend/app/core/database.py
git commit -m "feat: SQLAlchemy models (Document, ParentChunk) + async engine"
```

---

## Task 13: OpenAIEmbedder

**Files:**
- Create: `backend/app/pipeline/embedders/openai_embedder.py`
- Create: `tests/unit/pipeline/test_embedder.py`

**Step 1: Write test**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.pipeline.embedders.openai_embedder import OpenAIEmbedder


@pytest.fixture
def mock_openai():
    with patch("backend.app.pipeline.embedders.openai_embedder.AsyncOpenAI") as mock:
        client = AsyncMock()
        mock.return_value = client

        embedding_item = MagicMock()
        embedding_item.embedding = [0.1] * 1536

        response = MagicMock()
        response.data = [embedding_item]
        client.embeddings.create = AsyncMock(return_value=response)

        yield client


async def test_embed_single(mock_openai):
    embedder = OpenAIEmbedder()
    result = await embedder.embed_single("test text")
    assert len(result) == 1536


async def test_embed_batch(mock_openai):
    embedder = OpenAIEmbedder()
    # Mock returns one item per call, but we test the batch logic
    embedding_items = [MagicMock(embedding=[0.1] * 1536) for _ in range(3)]
    response = MagicMock()
    response.data = embedding_items
    mock_openai.embeddings.create = AsyncMock(return_value=response)

    result = await embedder.embed(["text1", "text2", "text3"])
    assert len(result) == 3
    assert all(len(v) == 1536 for v in result)


async def test_embed_empty_list(mock_openai):
    embedder = OpenAIEmbedder()
    result = await embedder.embed([])
    assert result == []


def test_dimensions():
    with patch("backend.app.pipeline.embedders.openai_embedder.AsyncOpenAI"):
        embedder = OpenAIEmbedder()
        assert embedder.dimensions == 1536
```

**Step 2: Implement `backend/app/pipeline/embedders/openai_embedder.py`**

```python
import asyncio

from openai import AsyncOpenAI

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.pipeline.base.embedder import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):

    def __init__(self):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model
        self._dimensions = settings.embedding_dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        batch_size = 100
        all_vectors = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = await self._embed_batch_with_retry(batch)
            all_vectors.extend(vectors)

        return all_vectors

    async def embed_single(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        return vectors[0]

    async def _embed_batch_with_retry(
        self,
        texts: list[str],
        max_retries: int = 3,
    ) -> list[list[float]]:
        for attempt in range(max_retries):
            try:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2**attempt
                logger.warning(
                    "embedding_retry", attempt=attempt, wait=wait, error=str(e)
                )
                await asyncio.sleep(wait)
```

**Step 3: Run tests**

```bash
uv run pytest tests/unit/pipeline/test_embedder.py -v
```

**Step 4: Commit**

```bash
git add backend/app/pipeline/embedders/openai_embedder.py tests/unit/pipeline/test_embedder.py
git commit -m "feat: OpenAIEmbedder with batch support and retry"
```

---

## Task 14: QdrantWrapper

**Files:**
- Create: `backend/app/vectorstore/qdrant_client.py`

**Step 1: Implement**

```python
import uuid

from qdrant_client import QdrantClient as _QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from backend.app.core.config import settings
from backend.app.core.exceptions import VectorStoreError
from backend.app.core.logging import logger


class QdrantWrapper:

    def __init__(self):
        self._client = _QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self._collection = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dimensions,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("qdrant_collection_created", collection=self._collection)

    async def upsert(self, chunks: list, vectors: list[list[float]]):
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=chunk.qdrant_payload(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        batch_size = 100
        for i in range(0, len(points), batch_size):
            try:
                self._client.upsert(
                    collection_name=self._collection,
                    points=points[i : i + batch_size],
                )
            except Exception as e:
                raise VectorStoreError(f"Qdrant upsert failed: {e}") from e

        logger.info("qdrant_upsert_done", count=len(points))

    async def search(
        self,
        vector: list[float],
        top_k: int = 20,
        score_threshold: float = None,
        filters: dict = None,
    ) -> list[ScoredPoint]:
        qdrant_filter = self._build_filter(filters) if filters else None

        try:
            results = self._client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=top_k,
                score_threshold=score_threshold or settings.retrieval_score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        except Exception as e:
            raise VectorStoreError(f"Qdrant search failed: {e}") from e

        return results

    async def delete_by_doc_id(self, doc_id: str):
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                ]
            ),
        )
        logger.info("qdrant_delete_done", doc_id=doc_id)

    def _build_filter(self, filters: dict) -> Filter:
        conditions = []

        if "doc_ids" in filters and filters["doc_ids"]:
            conditions.append(
                FieldCondition(
                    key="doc_id", match=MatchAny(any=filters["doc_ids"])
                )
            )
        if "language" in filters:
            conditions.append(
                FieldCondition(
                    key="language", match=MatchValue(value=filters["language"])
                )
            )
        if "type" in filters:
            conditions.append(
                FieldCondition(
                    key="type", match=MatchValue(value=filters["type"])
                )
            )

        return Filter(must=conditions) if conditions else None
```

**Step 2: Commit**

```bash
git add backend/app/vectorstore/qdrant_client.py
git commit -m "feat: QdrantWrapper with upsert, search, delete, filtering"
```

---

## Task 15: MetadataExtractor

**Files:**
- Create: `backend/app/pipeline/parsers/metadata_extractor.py`

**Step 1: Implement**

```python
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
```

**Step 2: Commit**

```bash
git add backend/app/pipeline/parsers/metadata_extractor.py
git commit -m "feat: MetadataExtractor — LLM-based doc-level metadata"
```

---

## Task 16: IngestionService

**Files:**
- Create: `backend/app/services/ingestion.py`

**Step 1: Implement**

```python
"""
Orchestrates the full ingestion pipeline:
preprocess → parse → normalize → extract_metadata → chunk → enrich → filter → embed → store
"""
import os
import uuid

from backend.app.core.database import AsyncSessionLocal
from backend.app.core.exceptions import IngestionError
from backend.app.core.logging import logger
from backend.app.models.document import Document, ParentChunk
from backend.app.pipeline.chunkers.enricher import ContextEnricher
from backend.app.pipeline.chunkers.quality_filter import QualityFilter
from backend.app.pipeline.chunkers.smart_router import SmartRouter
from backend.app.pipeline.embedders.openai_embedder import OpenAIEmbedder
from backend.app.pipeline.parsers.factory import ParserFactory
from backend.app.pipeline.parsers.metadata_extractor import MetadataExtractor
from backend.app.pipeline.parsers.preprocessor import PDFPreprocessor
from backend.app.vectorstore.qdrant_client import QdrantWrapper


class IngestionService:

    def __init__(self):
        self._preprocessor = PDFPreprocessor()
        self._metadata_extractor = MetadataExtractor()
        self._router = SmartRouter()
        self._enricher = ContextEnricher()
        self._quality_filter = QualityFilter()
        self._embedder = OpenAIEmbedder()
        self._qdrant = QdrantWrapper()

    async def ingest(
        self,
        file_path: str,
        doc_name: str,
        language: str = "en",
        parser_strategy: str = "auto",
    ) -> dict:
        doc_id = str(uuid.uuid4())[:12]
        log = logger.bind(doc_id=doc_id, doc_name=doc_name)
        log.info("ingestion_start")

        try:
            # 1. Pre-process
            log.info("stage_preprocess")
            clean_path = self._preprocessor.preprocess(file_path)

            # 2. Parse
            log.info("stage_parse", strategy=parser_strategy)
            if parser_strategy == "auto":
                parser = ParserFactory.auto_select(clean_path)
            else:
                parser = ParserFactory.create(parser_strategy)

            elements = await parser.parse(
                clean_path, doc_id=doc_id, doc_name=doc_name
            )
            log.info(
                "stage_parse_done",
                elements=len(elements),
                parser=type(parser).__name__,
            )

            if not elements:
                raise IngestionError("No content extracted from document")

            detected_lang = next(
                (el.language for el in elements if el.language), language
            )

            # 3. Extract doc-level metadata
            log.info("stage_metadata")
            doc_metadata = await self._metadata_extractor.extract(elements)
            doc_metadata.update(
                {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "language": detected_lang,
                    "parser": type(parser).__name__,
                }
            )

            # 4. Chunk
            log.info("stage_chunk")
            parent_chunks, child_chunks = self._router.route(
                elements, doc_metadata
            )
            log.info(
                "stage_chunk_done",
                parents=len(parent_chunks),
                children=len(child_chunks),
            )

            # 5. Enrich child chunks
            log.info("stage_enrich")
            enriched_chunks = await self._enricher.enrich_batch(
                [c for c in child_chunks if not c.is_parent],
                doc_metadata,
            )

            # 6. Quality filter
            final_children = self._quality_filter.filter(enriched_chunks)
            log.info("stage_enrich_done", after_filter=len(final_children))

            # 7. Embed children
            log.info("stage_embed")
            texts_to_embed = [c.content for c in final_children]
            vectors = await self._embedder.embed(texts_to_embed)

            # 8. Store in Qdrant (children)
            log.info("stage_store_qdrant")
            await self._qdrant.upsert(final_children, vectors)

            # 9. Store parents in PostgreSQL
            log.info("stage_store_postgres")
            await self._store_parents(
                doc_id, doc_name, parent_chunks, doc_metadata
            )

            result = {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "language": detected_lang,
                "parser_used": type(parser).__name__,
                "elements_parsed": len(elements),
                "parent_chunks": len(parent_chunks),
                "child_chunks": len(final_children),
                "doc_type": doc_metadata.get("doc_type", "document"),
            }
            log.info("ingestion_done", **result)
            return result

        except Exception as e:
            log.error("ingestion_failed", error=str(e))
            raise IngestionError(f"Ingestion failed for {doc_name}: {e}") from e
        finally:
            if clean_path != file_path and os.path.exists(clean_path):
                os.unlink(clean_path)

    async def _store_parents(
        self,
        doc_id: str,
        doc_name: str,
        parent_chunks: list,
        doc_metadata: dict,
    ):
        async with AsyncSessionLocal() as session:
            # Store document record
            doc = Document(
                doc_id=doc_id,
                doc_name=doc_name,
                language=doc_metadata.get("language", "en"),
                doc_type=doc_metadata.get("doc_type", "document"),
                chunk_count=len(parent_chunks),
                parser_used=doc_metadata.get("parser", ""),
                status="ready",
                metadata_=doc_metadata,
            )
            session.add(doc)

            for chunk in parent_chunks:
                pg_chunk = ParentChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=doc_id,
                    content_raw=chunk.content_raw,
                    content_markdown=chunk.content_markdown,
                    content_html=chunk.content_html,
                    type=chunk.type,
                    page=chunk.page,
                    section=chunk.section,
                    language=chunk.language,
                    word_count=chunk.word_count,
                    metadata_=doc_metadata,
                )
                session.add(pg_chunk)
            await session.commit()

    async def delete_document(self, doc_id: str):
        await self._qdrant.delete_by_doc_id(doc_id)
        async with AsyncSessionLocal() as session:
            from sqlalchemy import delete

            await session.execute(
                delete(ParentChunk).where(ParentChunk.doc_id == doc_id)
            )
            await session.execute(
                delete(Document).where(Document.doc_id == doc_id)
            )
            await session.commit()
```

**Step 2: Commit**

```bash
git add backend/app/services/ingestion.py
git commit -m "feat: IngestionService — full pipeline orchestration"
```

---

## Task 17: Celery Setup + Tasks

**Files:**
- Create: `backend/app/workers/celery_app.py`
- Create: `backend/app/workers/ingest_tasks.py`

**Step 1: Create `backend/app/workers/celery_app.py`**

```python
from celery import Celery

from backend.app.core.config import settings

celery_app = Celery(
    "docmind",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    task_track_started=True,
    task_soft_time_limit=600,
    task_time_limit=900,
)
```

**Step 2: Create `backend/app/workers/ingest_tasks.py`**

```python
import asyncio

from backend.app.core.logging import logger
from backend.app.services.ingestion import IngestionService
from backend.app.workers.celery_app import celery_app


@celery_app.task(bind=True, name="ingest_document")
def ingest_document_task(
    self,
    file_path: str,
    doc_name: str,
    language: str = "en",
    parser_strategy: str = "auto",
):
    """Async ingestion task — runs in Celery worker, not HTTP request."""
    try:
        service = IngestionService()
        result = asyncio.run(
            service.ingest(file_path, doc_name, language, parser_strategy)
        )
        return result
    except Exception as e:
        logger.error(
            "celery_ingest_failed", task_id=self.request.id, error=str(e)
        )
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise
```

**Step 3: Commit**

```bash
git add backend/app/workers/
git commit -m "feat: Celery app + ingest_document task"
```

---

## Task 18: FastAPI Endpoints + main.py

**Files:**
- Create: `backend/app/api/documents.py`
- Create: `backend/app/api/health.py`
- Create: `backend/app/main.py`

**Step 1: Create `backend/app/api/health.py`**

```python
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "healthy"}
```

**Step 2: Create `backend/app/api/documents.py`**

```python
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.app.core.logging import logger
from backend.app.workers.ingest_tasks import ingest_document_task

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    language: str = Form(default="en"),
    parser_strategy: str = Form(default="auto"),
):
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size: 50MB",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    logger.info(
        "upload_received",
        filename=file.filename,
        size_mb=round(len(content) / 1024 / 1024, 2),
    )

    task = ingest_document_task.delay(
        file_path=tmp_path,
        doc_name=file.filename,
        language=language,
        parser_strategy=parser_strategy,
    )

    return JSONResponse(
        {
            "status": "processing",
            "task_id": task.id,
            "message": f"Document '{file.filename}' queued for processing",
        }
    )


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task = ingest_document_task.AsyncResult(task_id)
    if task.state == "SUCCESS":
        return {"status": "ready", "result": task.result}
    elif task.state == "FAILURE":
        return {"status": "failed", "error": str(task.result)}
    else:
        return {"status": task.state.lower()}


@router.get("/")
async def list_documents():
    from sqlalchemy import select

    from backend.app.core.database import AsyncSessionLocal
    from backend.app.models.document import Document

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Document))
        docs = result.scalars().all()
    return {
        "documents": [
            {"doc_id": d.doc_id, "doc_name": d.doc_name, "status": d.status}
            for d in docs
        ],
        "total": len(docs),
    }


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    from backend.app.services.ingestion import IngestionService

    service = IngestionService()
    await service.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}
```

**Step 3: Create `backend/app/main.py`**

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import documents, health
from backend.app.core.database import create_tables
from backend.app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    await create_tables()
    yield


app = FastAPI(
    title="DocMind RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    documents.router, prefix="/api/v1/documents", tags=["documents"]
)
app.include_router(health.router, prefix="/api/v1", tags=["health"])
```

**Step 4: Commit**

```bash
git add backend/app/api/ backend/app/main.py
git commit -m "feat: FastAPI endpoints — upload, task status, list, delete, health"
```

---

## Task 19: Integration Test

**Files:**
- Create: `tests/integration/test_ingestion_pipeline.py`
- Create: `tests/conftest.py`

**Step 1: Create `tests/conftest.py`**

```python
import os

import pytest


@pytest.fixture
def sample_pdf_path():
    path = "tests/fixtures/sample.pdf"
    if not os.path.exists(path):
        pytest.skip("Test PDF not found at tests/fixtures/sample.pdf")
    return path
```

**Step 2: Create `tests/integration/test_ingestion_pipeline.py`**

```python
"""
Requires: Qdrant + PostgreSQL running (docker compose up qdrant postgres)
Run with: pytest tests/integration/ -m integration
"""
import os

import pytest


@pytest.mark.integration
async def test_full_ingestion_pipeline(sample_pdf_path):
    from backend.app.services.ingestion import IngestionService

    service = IngestionService()
    result = await service.ingest(
        file_path=sample_pdf_path,
        doc_name="sample.pdf",
        language="en",
    )

    assert result["doc_id"] is not None
    assert result["child_chunks"] > 0
    assert result["parent_chunks"] > 0
    assert result["child_chunks"] >= result["parent_chunks"]


@pytest.mark.integration
async def test_delete_removes_from_stores(sample_pdf_path):
    from backend.app.services.ingestion import IngestionService
    from backend.app.vectorstore.qdrant_client import QdrantWrapper

    service = IngestionService()
    result = await service.ingest(sample_pdf_path, "test.pdf")
    doc_id = result["doc_id"]

    await service.delete_document(doc_id)

    qdrant = QdrantWrapper()
    search_results = await qdrant.search(
        vector=[0.0] * 1536,
        filters={"doc_ids": [doc_id]},
    )
    assert len(search_results) == 0
```

**Step 3: Commit**

```bash
git add tests/
git commit -m "test: integration tests for full ingestion pipeline"
```

---

## Task 20: Final Verification

**Step 1: Run all unit tests**

```bash
uv run pytest tests/unit/ -v
```

Expected: All pass.

**Step 2: Start infrastructure**

```bash
docker compose up -d
```

**Step 3: Verify structlog output**

```bash
OPENAI_API_KEY=test uv run python -c "
from backend.app.core.logging import configure_logging, logger
configure_logging()
logger.info('test_message', doc_id='abc', parser='docling')
"
```

Expected: JSON output with timestamp, level, doc_id.

**Step 4: Start FastAPI (smoke test)**

```bash
OPENAI_API_KEY=test uv run uvicorn backend.app.main:app --port 8000
# In another terminal: curl http://localhost:8000/api/v1/health
```

Expected: `{"status": "healthy"}`

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: Week 1 complete — ingestion pipeline end-to-end"
```
