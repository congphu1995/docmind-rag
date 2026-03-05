# DocMind RAG

Upload documents, parse, chunk, embed, and retrieve with augmented generation.

## Architecture

```
PDF/DOCX --> Preprocessor --> Parser --> Normalizer --> MetadataExtractor (LLM)
  --> SmartRouter --> ParentChildChunker --> ContextEnricher (LLM)
  --> QualityFilter --> Embedder --> Qdrant + PostgreSQL
```

**Stack:** FastAPI, Celery, SQLAlchemy (async), Qdrant, PostgreSQL, Redis, React + Vite + shadcn/ui

### Key Design Decisions

- **Strategy pattern** — parsers, chunkers, embedders, retrievers, rerankers all implement base ABCs
- **Parent-child chunking** — parents (~800 words) in PostgreSQL for LLM context; children (~150 words) in Qdrant for vector search
- **Agentic RAG** — LangGraph agent with query analysis, retrieval, reranking, and generation nodes
- **Async everywhere** — FastAPI + asyncpg + async SQLAlchemy; Celery bridges sync-to-async

### Project Structure

```
backend/
  app/
    api/              # FastAPI routes (documents, chat, eval, health)
    agent/            # LangGraph RAG agent (graph, nodes, state)
    core/             # Config, database, logging, exceptions
    models/           # SQLAlchemy models (Document, ParentChunk)
    pipeline/
      parsers/        # DoclingParser, PyMuPDFParser, Preprocessor, Normalizer
      chunkers/       # SmartRouter, ParentChildChunker, QualityFilter
      embedders/      # OpenAIEmbedder
      rerankers/      # Reranker implementations
      llm/            # LLM client abstractions
      multimodal/     # Multimodal processing
    services/         # IngestionService (pipeline orchestrator)
    vectorstore/      # QdrantWrapper
    workers/          # Celery app + async ingest task
frontend/             # React + Vite + TypeScript + shadcn/ui + Zustand
eval/                 # Evaluation datasets, notebooks, results
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Node.js 18+
- Docker & Docker Compose

### 1. Clone and install

```bash
git clone https://github.com/congphu1995/docmind-rag.git
cd docmind-rag
uv sync
cd frontend && npm install && cd ..
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 3. Start services

```bash
# Start infrastructure (Qdrant, PostgreSQL, Redis)
make infra

# In separate terminals:
make backend    # FastAPI on :8000
make worker     # Celery worker
make frontend   # Vite dev server on :5173
```

Or run the full stack with Docker:

```bash
docker compose up
```

### 4. Open the app

- Frontend: http://localhost:5173 (dev) or http://localhost:3000 (Docker)
- API docs: http://localhost:8000/docs

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make infra` | Start Docker services (Qdrant, PostgreSQL, Redis) |
| `make infra-down` | Stop Docker services |
| `make backend` | Run FastAPI server with hot reload |
| `make worker` | Run Celery worker |
| `make frontend` | Run Vite dev server |
| `make dev` | Start infra + print instructions |
| `make test` | Run unit tests |
| `make test-all` | Run all tests |
| `make lint` | Check linting (ruff + format) |
| `make lint-fix` | Auto-fix lint issues |
| `make eval` | Download evaluation dataset |
| `make seed` | Seed demo data |
| `make clean` | Remove Python cache files |

## Environment Variables

See [`.env.example`](.env.example) for all variables. `OPENAI_API_KEY` is required.

## License

MIT
