# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git
- Do not add `Co-Authored-By` lines to commit messages.

## Commands

```bash
# Dependencies (uses uv, not pip)
uv sync                              # Install all deps
uv sync --group dev                  # Include dev deps

# Run app
uvicorn backend.app.main:app --reload

# Celery worker
celery -A backend.app.workers.celery_app worker --loglevel=info

# Tests
pytest                               # All tests
pytest tests/unit                    # Unit tests only
pytest tests/unit/pipeline/test_chunkers.py::test_parent_word_count -v  # Single test
pytest tests/integration -m integration  # Integration (requires Docker services)

# Infrastructure
docker compose up -d                 # Start Qdrant, PostgreSQL, Redis

# Linting
ruff check .
ruff format .
mypy backend/
```

## Architecture

**DocMind RAG** — Upload documents → parse → chunk → embed → store in Qdrant + PostgreSQL for retrieval-augmented generation.

### Pipeline Flow (IngestionService)

```
PDF/DOCX → PDFPreprocessor (deskew) → ParserFactory.auto_select() → ElementNormalizer
  → MetadataExtractor (LLM) → SmartRouter → ParentChildChunker
  → ContextEnricher (LLM) → QualityFilter → OpenAIEmbedder
  → Qdrant (child vectors) + PostgreSQL (parent chunks + metadata)
```

### Strategy Pattern — All Components Behind ABCs

- `pipeline/base/parser.py` → `BaseParser` — implementations: `DoclingParser`, `PyMuPDFParser`
- `pipeline/base/chunker.py` → `BaseChunker` — implementation: `ParentChildChunker`
- `pipeline/base/embedder.py` → `BaseEmbedder` — implementation: `OpenAIEmbedder`
- `pipeline/base/retriever.py` → `BaseRetriever` — implementation: `AdaptiveRetriever`
- `pipeline/base/reranker.py` → `BaseReranker` — implementation: `IdentityReranker`

### Semantic Parent-Child Chunking

**Parents** are section-based (one heading = one parent). Sections < 200 words are merged, > 1200 words split at paragraph boundaries. Target ~800 words. Stored in PostgreSQL.

**Children** are paragraph-based within each parent. Paragraphs < 50 words are merged, > 250 words split at sentence boundaries (via `SentenceSplitter`). Target ~150 words. Embedded in Qdrant.

Atomic elements (tables, figures, code) are never split. Titles mark section boundaries. Sentence-based overlap between parent groups.

### Key Directories

- `backend/app/core/` — Config (pydantic-settings), database (async SQLAlchemy), logging (structlog), exceptions
- `backend/app/pipeline/parsers/` — DoclingParser, PyMuPDFParser, ParserFactory (auto-selects by file analysis), Preprocessor, Normalizer, MetadataExtractor
- `backend/app/pipeline/chunkers/` — SmartRouter, ParentChildChunker, SentenceSplitter, QualityFilter, ContextEnricher
- `backend/app/services/ingestion.py` — Pipeline orchestrator (9 stages)
- `backend/app/workers/` — Celery app + async ingest task
- `backend/app/vectorstore/qdrant_client.py` — QdrantWrapper (upsert, search, delete, filtering)
- `backend/app/models/document.py` — Document + ParentChunk SQLAlchemy models

### Async Everywhere

FastAPI + SQLAlchemy async + asyncpg. Celery tasks bridge sync→async via `asyncio.run()`. All service methods are `async def`.

### Configuration

Pydantic Settings loaded from `.env` file. See `.env.example` for required vars. `OPENAI_API_KEY` is the only required key. Module-level singleton: `from backend.app.core.config import settings`.

### Docker Services

PostgreSQL on 5432 (user/pass/db: docmind), Qdrant on 6333, Redis on 6379.
