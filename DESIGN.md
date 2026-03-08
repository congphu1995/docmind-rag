# DocMind RAG Platform — System Design

> Production-grade, modular document intelligence platform.  
> Every component is swappable via config. Zero business logic changes to switch parsers, LLMs, chunkers, or vector stores.

| Field | Value |
|---|---|
| Author | Cong Phu Nguyen — Senior AI Engineer |
| Stack | FastAPI · LangGraph · Docling · Qdrant · React |
| Purpose | Portfolio project demonstrating production RAG architecture |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Component Design — Strategy Pattern](#3-component-design)
4. [Data Models & Contracts](#4-data-models--contracts)
5. [Folder Structure](#5-folder-structure)
6. [Infrastructure & Observability](#6-infrastructure--observability)
7. [Evaluation Strategy](#7-evaluation-strategy)
8. [What's Built & Roadmap](#8-whats-built--roadmap)
9. [Design Principles Reference](#9-design-principles-reference)

---

## 1. Project Overview

### 1.1 Core Design Philosophy

> Every component sits behind an abstract interface. Changing from Docling to Azure Document Intelligence, Qdrant to Pinecone, or Claude to GPT-4o requires only a config change — zero business logic changes.

### 1.2 Capability Matrix

| Capability | Implementation | Demo Value |
|---|---|---|
| Smart Parsing | Docling (default) + pymupdf4llm (fast fallback) | Handles real enterprise docs including scanned |
| Advanced Chunking | Structural Parent-Child + Semantic children + element routing | Shows production chunking depth |
| Agentic Retrieval | LangGraph: query classify → conditional HyDE → adaptive retry | Differentiates from basic RAG |
| Multimodal RAG | GPT-4o Vision for figures, NL + markdown + HTML for tables | Unique from CV background |
| Multi-LLM | Claude + GPT-4o switchable per query, identical interface | Flexibility clients need |
| Evaluation | RAGAS + FinanceBench with reproducible notebooks | Proves quality with real numbers |
| Observability | Langfuse LLM traces + structured logs with trace_id | Production credibility |

### 1.3 What We Build vs Skip

#### Must Build ✅

```
Parser:    DoclingParser, PyMuPDFParser, PDFPreprocessor, ElementNormalizer
Chunker:   SmartRouter, ParentChildChunker, ContextEnricher, QualityFilter
Agent:     query_analyzer, router, retriever (adaptive), generator, conditional HyDE
Infra:     Qdrant, PostgreSQL (parent chunks), Docker Compose
Eval:      RAGAS, FinanceBench, committed results JSON
Frontend:  Chat UI (streaming + citations + LLM toggle + agent trace), Upload zone
```

#### Roadmap ⏭

```
AzureDIParser                 roadmap — you have the Azure certs already
BGE-M3 local embedder         roadmap — multilingual support
SelfCheckNode                 add after core pipeline is stable
Redis query cache             add after basic pipeline works
```

---

## 2. System Architecture

### 2.1 Layer Diagram

```
┌──────────────────────────────────────────────────────────────┐
│  PRESENTATION LAYER                                          │
│  React SPA · Chat UI · Document Manager · Agent Trace Viewer │
└──────────────────────────┬───────────────────────────────────┘
                           │ REST + SSE
┌──────────────────────────▼───────────────────────────────────┐
│  API GATEWAY LAYER                                           │
│  FastAPI · Rate Limiting · Request Validation                │
│  Auto OpenAPI docs at /docs                                  │
└──────────────────────────┬───────────────────────────────────┘
                           │ Service interfaces
┌──────────────────────────▼───────────────────────────────────┐
│  SERVICE LAYER                                               │
│  IngestionService · RAGService · EvalService                 │
└──────────────────────────┬───────────────────────────────────┘
              Abstract base classes (Strategy Pattern)
┌──────────────────────────▼───────────────────────────────────┐
│  PIPELINE LAYER                                              │
│  Parsers · Chunkers · Embedders · LLM Clients                │
│  Retrievers · Rerankers · Agent Nodes                        │
└──────────────────────────┬───────────────────────────────────┘
                           │ Infrastructure clients
┌──────────────────────────▼───────────────────────────────────┐
│  INFRASTRUCTURE LAYER                                        │
│  Qdrant · PostgreSQL · MinIO · Celery · Langfuse             │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Ingestion Pipeline

```
Raw File (PDF / DOCX / TXT)
        │
        ▼
┌───────────────────┐
│  1. Pre-process   │  Detect: scanned, rotated, encrypted, corrupt
│  PDFPreprocessor  │  Repair if needed. Output: cleaned file path.
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  2. Parse         │  Docling (default) → pymupdf4llm (fast fallback)
│  ParserFactory    │  auto_select() based on file type + scan detection
└────────┬──────────┘  Output: List[RawParserOutput]
         │
         ▼
┌───────────────────┐
│  3. Normalize     │  Every parser → unified ParsedElement schema
│  ElementNormalizer│  Downstream code never knows which parser ran
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  4. Enrich        │  (a) Doc-level: extract title, doc_type, language, date
│  MetadataExtractor│      1 LLM call per document (gpt-4o-mini)
│  ContextEnricher  │  (b) Per-chunk: prepend 1-2 situating sentences before embed
└────────┬──────────┘      "This chunk is from [section] of [doc], describing [topic]."
         │                 One-time cost at index time. +15-20% retrieval precision.
         ▼
┌───────────────────┐
│  5. Chunk         │  SmartRouter decides per element type (see section 3.3)
│  SmartRouter      │  Parent chunks (800 words) → PostgreSQL
│  ParentChildChunker  Child chunks  (150 words) → Qdrant (embedded)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  6. Filter+Index  │  QualityFilter removes noise before embedding
│  QualityFilter    │  OpenAIEmbedder embeds child chunks
│  Embedder         │  Qdrant: vectors + full metadata payload
│  VectorStoreClient│  PostgreSQL: parent chunks keyed by parent_id
└───────────────────┘
```

### 2.3 Agentic Query Pipeline (LangGraph)

```
User Query
    │
    ▼
[query_analyzer]
  · Detect language
  · Classify: factual | analytical | tabular | multi_hop | general | greeting
  · Extract metadata filters (date, doc_type, etc.)
    │
    ▼
[router]─────────────────────────────────────────────┐
  · factual / analytical / tabular → query_rewriter   │
  · multi_hop                      → decomposer       │
  · general knowledge              → direct_llm       │
  · greeting                       → direct_response  │
    │                                                 │
    ▼                                                 │
[decomposer] (multi_hop only)                         │
  · Break into sub-questions                          │
    │                                                 │
    ▼                                                 │
[query_rewriter]                                      │
  · Always: expand abbreviations, resolve references  │
  · Conditional HyDE (see section 3.6):               │
    — Run: analytical, multi_hop, vague factual       │
    — Skip: specific factual, short keyword, follow-up│
  · HyDE model: gpt-4o-mini, max_tokens=150 (~300ms) │
    │                                                 │
    ▼                                                 │
[retriever]◄──── retry loop (max 3 attempts) ────┐   │
  · Dense search top-20 in Qdrant                 │   │
  · Metadata filter by doc_ids / language         │   │
  · Assess quality score                          │   │
  · Attempt 1: raw/hyde query                     │   │
  · Attempt 2: expanded query  (score < 0.6) ─────┘   │
  · Attempt 3: sub-question retrieval                 │
  · Fetch parent chunks from PostgreSQL               │
    │                                                 │
    ▼                                                 │
[reranker]                                            │
  · Identity (passthrough) by default                 │
  · Cohere Rerank v3 as roadmap upgrade               │
    │                                                 │
    ▼                                                 │
[generator]                                           │
  · Build prompt: enriched context + citations        │
  · Stream via SSE                                    │
  · Emit agent_trace (shown in UI)                    │
    │                                                 │
    ▼                                                 │
Answer + Sources + Confidence + Agent Trace           │
[direct_llm / direct_response]◄──────────────────────┘
```

**Latency budget:**

| Query Type | HyDE Used | Expected p50 to First Token |
|---|---|---|
| Simple factual | No | ~1.0s |
| Analytical / multi-hop | Yes (gpt-4o-mini, 150 tok) | ~1.4s |
| General knowledge | No retrieval | ~0.6s |

Target p95 < 3s. Both retrieval cases well within budget.

---

## 3. Component Design

### 3.1 Abstract Base Class Pattern

Every swappable component follows: **ABC → Concrete Implementation → Factory**

```python
# pipeline/base/parser.py
class BaseParser(ABC):
    @abstractmethod
    async def parse(self, file_path: str, **kwargs) -> list[ParsedElement]: ...
    def supports(self, ext: str) -> bool: ...
    def get_capabilities(self) -> ParserCapabilities: ...

# Business logic NEVER imports a concrete class:
parser = ParserFactory.create(settings.parser_strategy)
# Returns: DoclingParser | PyMuPDFParser
# Switching parser = change .env, redeploy. Zero code changes.
```

Same pattern for: `BaseChunker`, `BaseEmbedder`, `BaseLLMClient`, `BaseRetriever`, `BaseReranker`.

---

### 3.2 Parser Layer

| Provider | Role | Scanned? | Tables | Speed |
|---|---|---|---|---|
| **Docling** | Default — all types | ✓ built-in OCR | Excellent | Medium |
| **pymupdf4llm** | Fast fallback — clean born-digital PDFs | ✗ | Good | Fast |
| Azure Doc Intelligence | Roadmap — forms, invoices | ✓ best-in-class | Excellent | Slow |
| Unstructured | Roadmap — DOCX, PPTX, HTML | ✓ hi_res | Good | Medium |

**Auto-selection logic:**
```python
# ParserFactory.auto_select(file_path)
# Scanned PDF    → DoclingParser  (OCR enabled)
# Clean PDF      → PyMuPDFParser  (fast path)
# DOCX / TXT     → DoclingParser  (handles natively)
# Fallback       → DoclingParser
```

**Pre-processing (always runs before parsing):**
```python
class PDFPreprocessor:
    def diagnose(self, file_path) -> dict:
        return {
            "is_scanned":   text_density < 0.001,
            "is_rotated":   page.rotation != 0,
            "is_encrypted": doc.is_encrypted,
            "is_corrupt":   doc.is_repaired,
        }
    # is_rotated   → deskew before parsing
    # is_encrypted → raise clear error to user
    # is_corrupt   → attempt ghostscript repair
```

**Normalization — the most important step in the parser layer:**
`ElementNormalizer` converts all parser outputs to `ParsedElement` before any downstream stage. This is what makes the strategy pattern work — downstream code is 100% parser-agnostic.

---

### 3.3 Chunker Layer

**Semantic Parent-Child chunking — paragraph-aware, section-based:**

```
Parents: section-based (one heading = one parent)
  Merge sections < 200 words. Split sections > 1200 words at paragraph boundaries.
  Target ~800 words. Stored in PostgreSQL.

Children: paragraph-based within each parent
  Merge paragraphs < 50 words. Split paragraphs > 250 words at sentence boundaries.
  Target ~150 words. Embedded and stored in Qdrant.

Sentence-based overlap between parent groups (last sentence of previous child).
```

**SmartRouter — element type determines strategy:**

| Element Type | Strategy | Why |
|---|---|---|
| **Table** | Always atomic | Split table = meaningless. Biggest single retrieval impact. |
| **Figure** | Always atomic | Vision description is already one coherent unit |
| **Code** | Always atomic | Split code = broken code |
| **Title / Heading** | Not chunked — section boundary | Marks where a new parent starts |
| **Text / List / Scanned** | Semantic Parent-Child | See below |

**Text routing:**
```
Has headings detected?
  Yes → heading boundaries as parents (structural)
        paragraph-based children within each section
  No  → paragraph-based parents (split at paragraph boundaries)
        paragraph-based children within each parent

SentenceSplitter handles oversized paragraphs — splits at sentence
boundaries respecting abbreviations (Dr., e.g., D.C.), punctuation,
and newlines.
```

**Parent-Child explained:**
```
Parent chunk (~800 words) — stored in PostgreSQL
  One full document section with all context.
  What gets sent to the LLM.

  ├── Child A (paragraph 1, ~120 words) — stored in Qdrant (embedded)
  ├── Child B (paragraphs 2-3 merged, ~90 words) — stored in Qdrant
  └── Child C (paragraph 4, ~180 words) — stored in Qdrant
        Natural paragraph boundaries. Preserves document structure.

Query time:
  1. Embed query → search Qdrant → Child B scores highest
  2. Look up Child B's parent_id → fetch parent from PostgreSQL
  3. Send PARENT (full ~800 words) to LLM, not just the child

Why: short children = precise retrieval.
     long parents = rich context for generation.
     You index small, you retrieve large.
```

**Configurable chunk size ranges:**

| Parameter | Default | Purpose |
|---|---|---|
| `parent_min_words` | 200 | Merge small sections below this |
| `parent_max_words` | 1200 | Split sections above this |
| `child_min_words` | 50 | Merge small paragraphs below this |
| `child_max_words` | 250 | Split paragraphs above this |

**Contextual enrichment (highest ROI step):**
```python
# Before embedding, prepend situating context to each child chunk.
# Cost: 1 LLM call per chunk at index time (one-off, gpt-4o-mini).
# Benefit: +15-20% retrieval precision.

# Before:
"Claims must be submitted within 90 days of service date."

# After enrichment:
"This chunk is from the Dental Benefits section of the 2024
 Employee Policy Manual, describing claim submission deadlines.

 Claims must be submitted within 90 days of service date."
```

**Quality filter (before indexing):**
```python
def is_quality_chunk(chunk) -> bool:
    if len(chunk.content.split()) < 15:       return False  # too short
    if alpha_ratio(chunk.content) < 0.4:      return False  # noise / page numbers
    if has_repeated_chars(chunk.content):     return False  # OCR artifacts
    if is_pure_numeric_table(chunk.content):  return False  # numbers without headers
    return True
```

**Table representation — store all three, use different ones at different stages:**
```python
{
  "natural_language": "Row 1: Q1 revenue is 12.4M, Q2 is 15.1M...",  # embed this
  "markdown":         "| Q1 | Q2 |\n|12.4M|15.1M|",                  # send to LLM
  "html":             "<table><tr><td>12.4M</td>...",                  # store for UI
}
# Embed NL: matches how users phrase questions
# Send markdown to LLM: easier to reason about structure
```

---

### 3.4 Embedder Layer

| Provider | Model | Dimensions | Status |
|---|---|---|---|
| **OpenAI** (default) | text-embedding-3-small | 1536 | Built now |
| OpenAI large | text-embedding-3-large | 3072 | Config swap |
| BGE-M3 (local) | BAAI/bge-m3 | 1024 | Roadmap — multilingual |

---

### 3.5 LLM Provider Layer

```python
# pipeline/base/llm_client.py — identical interface, all providers
class BaseLLMClient(ABC):
    @abstractmethod
    async def complete(self, messages, system, **kwargs) -> str: ...
    @abstractmethod
    async def stream(self, messages, system, **kwargs) -> AsyncGenerator[str, None]: ...
    @abstractmethod
    async def complete_with_vision(self, messages, images, **kwargs) -> str: ...
```

| Provider | Model | Role |
|---|---|---|
| **GPT-4o** (primary) | gpt-4o | Main generation |
| **Claude** (secondary) | claude-sonnet-4-20250514 | Alternative — user toggles in UI |
| **GPT-4o-mini** | gpt-4o-mini | HyDE generation, metadata extraction, doc summary |
| **GPT-4o Vision** | gpt-4o | Figure description, scanned page OCR fallback |

All prompts live in dedicated files — versioned, with rationale comments. Never inline.
- `pipeline/prompts.py` — ingestion prompts (enrichment, metadata extraction)
- `agent/prompts.py` — query pipeline prompts (HyDE, generation, analysis)

LLM responses that return structured data use **OpenAI structured output** (`client.beta.chat.completions.parse()` with Pydantic models) instead of raw JSON parsing. Response schemas live in `schemas/pipeline.py`.

---

### 3.6 Conditional HyDE

HyDE is NOT always on. Decided inside `query_rewriter` based on `query_analyzer` output.

```python
def _should_use_hyde(self, query: str, query_type: str) -> bool:

    # Never use HyDE
    if query_type in ("greeting", "doc_management", "general"):
        return False
    if len(query.split()) <= 5:           # already specific
        return False
    if self._is_keyword_query(query):     # "Form D-14", "IBNR reserve"
        return False

    # Use HyDE
    if query_type == "multi_hop":         return True
    if query_type == "analytical":        return True
    if query_type == "factual" and len(query.split()) >= 10:
        return True                       # vague — user unsure how to phrase

    return False  # default: skip

# When HyDE runs:
#   model:      gpt-4o-mini   (quality doesn't matter, just needs good embedding)
#   max_tokens: 150           (enough for embedding, keeps latency ~300ms)
```

Agent trace always records the decision:
```python
"HyDE: skipped — query is 4 words, specific enough"
"HyDE: used — analytical query, generated 138 tokens in 290ms"
```

---

### 3.7 Reranker Layer

| Provider | Type | Status |
|---|---|---|
| **Identity** (default) | Passthrough — no reranking | Built |
| **Cohere Rerank v3** | API cross-encoder, top-20 → top-N | Built |
| BGE Reranker | Local cross-encoder | Roadmap |

Both rerankers implement `BaseReranker`. Switch via `RERANKER_STRATEGY` config (`identity` or `cohere`). `RerankerFactory.create()` handles instantiation.

---

## 4. Data Models & Contracts

### 4.1 Core Schemas

```python
# ── ParsedElement ─────────────────────────────────────────────
# Universal output — EVERY parser normalizes to this.
# Downstream code never imports a parser directly.

@dataclass
class ParsedElement:
    type: ElementType          # TEXT|TABLE|FIGURE|TITLE|LIST_ITEM|CODE|SCANNED
    content: str               # text or markdown representation
    page: int
    doc_id: str
    doc_name: str
    section_title: Optional[str] = None   # heading this element belongs to
    bbox: Optional[tuple] = None          # (x0, y0, x1, y1)
    reading_order: Optional[int] = None
    image_b64: Optional[str] = None       # raw image for figures
    table_html: Optional[str] = None
    table_df: Optional[pd.DataFrame] = None
    confidence: Optional[float] = None    # OCR confidence
    language: Optional[str] = None        # detected language
    is_scanned: bool = False
    parser_used: str = ""                 # audit trail


# ── Chunk ─────────────────────────────────────────────────────
# Children → Qdrant (embedded, for retrieval)
# Parents  → PostgreSQL (full context, returned to LLM)

@dataclass
class Chunk:
    chunk_id: str
    parent_id: Optional[str]   # None if this IS a parent
    doc_id: str
    doc_name: str
    content: str               # enriched — what gets embedded
    content_raw: str           # original — what LLM sees as context
    content_markdown: Optional[str]
    content_html: Optional[str]
    type: str                  # "text"|"table"|"figure"|"scanned"
    page: int
    section: str
    language: str
    word_count: int
    is_parent: bool
    metadata: dict             # extensible: doc_type, date, org...


# ── RAGAgentState ─────────────────────────────────────────────
# LangGraph typed state — flows through every node

class RAGAgentState(TypedDict):
    # Input
    original_query: str
    doc_ids: list[str]
    llm_preference: str              # "claude" | "openai"

    # Query understanding
    query_type: str                  # "factual"|"analytical"|"tabular"|"multi_hop"|"general"|"greeting"
    sub_questions: list[str]
    extracted_filters: dict
    detected_language: str

    # Query rewriting
    rewritten_query: str
    hyde_query: Optional[str]        # None if HyDE skipped
    hyde_used: bool                  # for analytics + trace

    # Retrieval
    retrieved_chunks: list[dict]
    reranked_chunks: list[dict]
    retrieval_attempts: int
    retrieval_quality: float

    # Generation
    answer: str
    citations: list[dict]

    # Observability
    messages: Annotated[list, add_messages]
    agent_trace: list[str]           # shown in UI
    langfuse_trace_id: str
    error: Optional[str]
```

### 4.2 API Contracts

| Endpoint | Method | Request | Response |
|---|---|---|---|
| `POST /api/v1/auth/register` | JSON | `email, username, password` | `{ access_token, refresh_token, user }` |
| `POST /api/v1/auth/login` | JSON | `email, password` | `{ access_token, refresh_token, user }` |
| `POST /api/v1/auth/refresh` | JSON | `refresh_token` | `{ access_token, refresh_token }` |
| `POST /api/v1/documents/upload` | multipart | `file`, `language?` | `{ doc_id, doc_name, task_id }` |
| `GET /api/v1/documents/` | — | — | `{ documents: [...], total }` |
| `GET /api/v1/documents/task/{taskId}` | — | — | `{ status, result }` |
| `GET /api/v1/documents/{docId}/chunks` | — | `type_filter?, page_filter?, search?` | `[{ chunk_id, content_raw, children, ... }]` |
| `DELETE /api/v1/documents/{id}` | — | — | `{ status, doc_id }` |
| `POST /api/v1/chat/` | JSON | `question, llm?, doc_ids?, history?, stream?` | SSE stream |
| `POST /api/v1/eval/run` | JSON | `dataset, sample_size?, config?` | `{ run_id }` async |
| `GET /api/v1/eval/results/{run_id}` | — | — | `{ metrics: {...} }` |
| `GET /api/v1/health` | — | — | `{ status, qdrant, postgres }` |
| `GET /metrics/` | — | — | Prometheus metrics (text) |

**SSE stream format:**
```
# Event 1: metadata
data: __META__{"sources":[...], "llm_used":"claude", "hyde_used":false, "agent_trace":[...]}__META__

# Events 2-N: streamed answer tokens
data: Based
data:  on
data:  the policy...

# Final
data: [DONE]
```

---

## 5. Folder Structure

```
docmind-rag/
│
├── DESIGN.md                       ← this file — read before coding anything
├── README.md                       ← public: architecture + eval + quickstart
├── docker-compose.yml
├── docker-compose.dev.yml          ← hot reload, verbose logging
├── Makefile                        ← make dev | test | eval | lint | seed
├── .env.example
│
├── backend/
│   ├── pyproject.toml              ← uv managed (not requirements.txt)
│   ├── Dockerfile
│   │
│   └── app/
│       ├── main.py
│       │
│       ├── api/                    ← HTTP routers ONLY — no business logic
│       │   ├── auth.py             ← JWT register/login/refresh
│       │   ├── documents.py
│       │   ├── chat.py
│       │   ├── chunks.py           ← chunk viewer API
│       │   ├── eval.py
│       │   ├── dependencies.py     ← get_current_user dependency
│       │   └── health.py
│       │
│       ├── core/                   ← cross-cutting, imported everywhere
│       │   ├── config.py           ← Pydantic Settings — ALL config here
│       │   ├── database.py         ← SQLAlchemy async engine + session
│       │   ├── exceptions.py       ← custom exception hierarchy
│       │   ├── logging.py          ← structlog + trace_id injection
│       │   ├── metrics.py          ← Prometheus counters + histograms
│       │   └── middleware.py       ← HTTP metrics middleware
│       │
│       ├── models/                 ← SQLAlchemy ORM
│       │   ├── document.py         ← Document, ParentChunk (parent store)
│       │   ├── user.py             ← User model (bcrypt hashed passwords)
│       │   └── eval.py             ← EvalRun, EvalResult
│       │
│       ├── schemas/                ← Pydantic contracts
│       │   ├── auth.py            ← TokenResponse, TokenUser, RegisterRequest
│       │   ├── document.py
│       │   ├── pipeline.py        ← structured output models (DocumentMetadata)
│       │   ├── chat.py
│       │   └── eval.py
│       │
│       ├── services/               ← business logic orchestration
│       │   ├── auth.py             ← JWT create/verify, password hashing
│       │   ├── ingestion.py        ← preprocess→parse→normalize→enrich→chunk→index
│       │   ├── rag.py              ← agent invocation + response streaming
│       │   └── eval.py             ← RAGAS runner + result storage
│       │
│       ├── pipeline/               ← all swappable components
│       │   ├── prompts.py          ← ingestion prompts (enrichment, metadata)
│       │   │
│       │   ├── base/               ← ABCs — import ONLY these in services/
│       │   │   ├── parser.py       ← BaseParser + ParserCapabilities + ParsedElement
│       │   │   ├── chunker.py      ← BaseChunker + Chunk
│       │   │   ├── embedder.py     ← BaseEmbedder
│       │   │   ├── llm_client.py   ← BaseLLMClient
│       │   │   ├── retriever.py    ← BaseRetriever
│       │   │   └── reranker.py     ← BaseReranker
│       │   │
│       │   ├── parsers/
│       │   │   ├── docling_parser.py
│       │   │   ├── pymupdf_parser.py
│       │   │   ├── preprocessor.py       ← diagnose + deskew + repair
│       │   │   ├── normalizer.py         ← any output → ParsedElement
│       │   │   └── factory.py            ← .create() + .auto_select()
│       │   │
│       │   ├── chunkers/
│       │   │   ├── parent_child_chunker.py ← semantic section/paragraph chunking
│       │   │   ├── sentence_splitter.py    ← sentence boundary detection
│       │   │   ├── smart_router.py       ← routes by element type
│       │   │   ├── enricher.py           ← contextual enrichment
│       │   │   ├── quality_filter.py     ← remove noise before indexing
│       │   │   └── factory.py
│       │   │
│       │   ├── embedders/
│       │   │   ├── openai_embedder.py
│       │   │   └── factory.py
│       │   │
│       │   ├── llm/
│       │   │   ├── claude_client.py
│       │   │   ├── openai_client.py
│       │   │   └── factory.py
│       │   │
│       │   ├── rerankers/
│       │   │   ├── identity_reranker.py  ← passthrough default
│       │   │   ├── cohere_reranker.py    ← Cohere Rerank v3
│       │   │   └── factory.py            ← identity | cohere via config
│       │   │
│       │   └── multimodal/
│       │       ├── figure_describer.py   ← GPT-4o Vision
│       │       ├── table_representer.py  ← NL + markdown + HTML
│       │       └── scanned_ocr.py        ← full-page OCR fallback
│       │
│       ├── agent/
│       │   ├── graph.py            ← StateGraph wiring + compile()
│       │   ├── state.py            ← RAGAgentState TypedDict
│       │   ├── prompts.py          ← ALL prompts, versioned with rationale
│       │   └── nodes/
│       │       ├── query_analyzer.py
│       │       ├── query_rewriter.py     ← conditional HyDE + expansion
│       │       ├── retriever.py          ← adaptive retry + parent fetch
│       │       ├── reranker.py
│       │       └── generator.py          ← streaming + citations
│       │
│       ├── vectorstore/
│       │   ├── qdrant_client.py    ← retry, batching, error handling
│       │   └── collections.py      ← schema + index management
│       │
│       └── workers/
│           ├── celery_app.py
│           └── ingest_tasks.py     ← async doc processing
│
├── frontend/
│   ├── package.json
│   └── src/
│       ├── pages/
│       │   ├── Login.tsx
│       │   ├── Chat.tsx
│       │   ├── Documents.tsx
│       │   └── Chunks.tsx
│       ├── components/
│       │   ├── chat/
│       │   │   ├── MessageBubble.tsx
│       │   │   ├── SourceCard.tsx     ← expandable, confidence + page ref
│       │   │   ├── AgentTrace.tsx     ← LangGraph decision steps
│       │   │   └── LLMToggle.tsx      ← Claude ↔ GPT-4o
│       │   ├── documents/
│       │   │   ├── UploadZone.tsx
│       │   │   └── DocCard.tsx
│       │   └── chunks/
│       │       ├── ChunkTree.tsx       ← parent/child tree view
│       │       └── ChunkDetail.tsx     ← metadata + content tabs
│       ├── stores/
│       │   ├── authStore.ts            ← Zustand — JWT tokens, user
│       │   ├── chatStore.ts            ← Zustand
│       │   ├── chunkStore.ts           ← Zustand — chunk selection
│       │   └── documentStore.ts
│       ├── hooks/
│       │   ├── useSSEChat.ts           ← SSE stream + META parsing
│       │   ├── useDocuments.ts
│       │   └── useChunks.ts
│       └── api/
│           ├── client.ts              ← axios + auth interceptor
│           ├── documents.ts
│           ├── chunks.ts
│           └── chat.ts
│
├── eval/
│   ├── README.md                  ← how to reproduce results
│   ├── datasets/
│   │   └── download_financebench.py
│   ├── results/
│   │   └── financebench_results.json   ← committed, reproducible
│   └── notebooks/
│       ├── 01_baseline_eval.ipynb      ← RAGAS on default config
│       └── 02_chunking_ablation.ipynb  ← fixed vs parent-child, tables split vs atomic
│
└── scripts/
    ├── seed_demo_data.py          ← load FinanceBench samples + pre-built index
    └── benchmark_parsers.py       ← Docling vs pymupdf speed + quality
```

---

## 6. Infrastructure & Observability

### 6.1 Services

| Service | Technology | Purpose | Port |
|---|---|---|---|
| Vector DB | Qdrant v1.9 | Child chunk vectors + metadata + filtered search | 6333 |
| Relational DB | PostgreSQL 16 | Parent chunks, document metadata, eval results | 5432 |
| Object Storage | MinIO | Raw uploaded files — S3-compatible | 9000 |
| Task Queue | Celery + Redis | Async ingestion — no HTTP timeout on large files | — |
| LLM Observability | Langfuse | Every LLM call: prompt, response, latency, cost | 3001 |
| Metrics | Prometheus | HTTP latency, LLM tokens, ingestion timing, retrieval | 9090 |
| Dashboards | Grafana | Prometheus data visualization | 3002 |
| Structured Logs | Structlog → stdout | JSON logs with trace_id across all stages | — |

Redis included as Celery broker. Query cache is roadmap.

### 6.2 Docker Compose Stack

```yaml
# docker compose up --build
# All services use TZ=Asia/Ho_Chi_Minh (set via env, no code changes)
services:
  qdrant       # :6333
  postgres     # :5432  (TZ + PGTZ = Asia/Ho_Chi_Minh)
  redis        # :6379  Celery broker
  minio        # :9000
  backend      # :8000  FastAPI (TZ from .env)
  worker       #        Celery (same image, CMD=celery worker)
  frontend     # :3000  React + Vite
  langfuse     # :3001  LLM traces
  prometheus   # :9090  Metrics scraping
  grafana      # :3002  Dashboards (admin/admin)
```

---

## 7. Evaluation Strategy

### 7.1 Primary Dataset — FinanceBench

**Why FinanceBench:**
- Finance domain = directly relevant to insurance clients
- Real 10-K / 10-Q filings — same doc type as enterprise clients use
- Mix of narrative + tables — exercises both chunking paths
- 10,000+ QA pairs — statistically meaningful numbers
- Questions need multi-step reasoning — exercises the agentic pipeline

**Chunker config for FinanceBench:**
```
Parent: 1000 words  (financial sections can be long)
Child:  180 words
Tables: always atomic  ← biggest single impact on this dataset
```

### 7.2 Target Metrics

| Metric | Tool | Definition | Target |
|---|---|---|---|
| Retrieval Hit Rate | Custom | % of questions where correct chunk in top-5 | > 80% |
| Faithfulness | RAGAS | % of answer claims grounded in context | > 0.85 |
| Answer Relevancy | RAGAS | Semantic similarity of answer to question | > 0.80 |
| Context Recall | RAGAS | % of ground truth covered by retrieved context | > 0.75 |
| Latency p95 | Logged | End-to-end time to first streamed token | < 3s |

### 7.3 Ablation Notebook

`02_chunking_ablation.ipynb` — the most impressive artifact for technical clients:

```
Config A: fixed-size (400 words, no parent-child, no enrichment)
Config B: parent-child (800/150, no enrichment)
Config C: parent-child + contextual enrichment
Config D: parent-child + enrichment + tables atomic   ← demo default

Expected: D > C > B > A
Tables-atomic gap (A→D) should be the largest single jump.
Shows every design decision is evidence-based, not arbitrary.
```

### 7.4 README Eval Table (Target Output)

```markdown
## Evaluation — FinanceBench (150 public company financial reports)

| Metric             | Score  |
|--------------------|--------|
| Retrieval Hit Rate | ~83%   |
| Faithfulness       | ~0.87  |
| Answer Relevancy   | ~0.82  |
| Context Recall     | ~0.79  |
| Latency p95        | ~1.4s  |

Parser: Docling | Chunker: Structural Parent-Child (800/150 words)
Embedder: text-embedding-3-small | LLM: Claude claude-opus-4-20250514
Reproduce: eval/notebooks/01_baseline_eval.ipynb
```

---

## 8. What's Built & Roadmap

### Built

| Layer | Components |
|---|---|
| **Parsing** | `DoclingParser`, `PyMuPDFParser`, `PDFPreprocessor`, `ElementNormalizer`, `ParserFactory` |
| **Chunking** | `SmartRouter`, `ParentChildChunker` (semantic section/paragraph), `SentenceSplitter`, `ContextEnricher`, `QualityFilter` |
| **Embedding** | `OpenAIEmbedder` (text-embedding-3-small) |
| **LLM** | `ClaudeClient`, `OpenAIClient`, `LLMFactory` — switchable per query |
| **Agent** | LangGraph: `query_analyzer`, `router`, `query_rewriter` (conditional HyDE), `AdaptiveRetriever` (retry loop), `IdentityReranker` + `CohereReranker`, `GeneratorNode` (SSE streaming + citations) |
| **Multimodal** | `FigureDescriber` (GPT-4o Vision), `TableRepresenter` (NL + markdown + HTML) |
| **Eval** | RAGAS + `EvalService` + FinanceBench dataset |
| **Frontend** | React Chat UI (streaming + citations + LLM toggle + agent trace), Document Manager, Upload Zone, Chunk Viewer |
| **Infra** | Qdrant, PostgreSQL, Redis/Celery, MinIO, Langfuse, Prometheus + Grafana, Docker Compose |
| **Auth** | JWT authentication (access + refresh tokens) |

### Roadmap

| Feature | Notes |
|---|---|
| `AzureDIParser` | Forms, invoices — Azure certs on CV |
| BGE-M3 local embedder | Multilingual support |
| Redis query cache | After basic pipeline is stable |
| FinanceBench ablation notebook | Chunking ablation with committed results |
| `SelfCheckNode` | Answer verification after core pipeline is stable |

---

## 9. Design Principles Reference

Every decision during implementation should map to one of these.

| Principle | Applied As | Anti-pattern Avoided |
|---|---|---|
| **Strategy Pattern** | Parser/Chunker/Embedder/LLM/Reranker behind ABCs. Swap via config. | Importing concrete classes in business logic |
| **Factory Pattern** | `ParserFactory.create(strategy)` — config drives instantiation | `if/else` chains scattered across codebase |
| **Single Responsibility** | `api/` = routing only. `services/` = orchestration. `pipeline/` = one thing each. | Fat service classes mixing concerns |
| **Fail Loud at Config** | Pydantic Settings validates at startup — missing key crashes immediately | Silent `None` causing confusing runtime errors |
| **Schema First** | `ParsedElement` + `Chunk` defined before any implementation | Each parser returning a different dict structure |
| **Observable by Default** | Every stage emits structured log with `trace_id` | Debugging with `print()` |
| **Test at the Seam** | Unit tests against ABCs. Integration tests against real Qdrant. | Tests tightly coupled to concrete implementations |
| **Async All the Way** | Every I/O is async: DB, Qdrant, LLM, file reads | Sync calls blocking FastAPI event loop |
| **Prompts as Config** | All prompts in `pipeline/prompts.py` + `agent/prompts.py`, versioned with rationale | Inline strings inside node logic |
| **Structured Output** | LLM responses parsed via Pydantic models (`client.beta.chat.completions.parse`) | Manual `json.loads()` with try/except |
| **No Layer Skipping** | API → Service → Pipeline only. Never API → Pipeline direct. | Route handlers calling Qdrant directly |
| **Decide Late** | HyDE, reranking are conditional — decided at runtime not hardcoded | Paying latency cost on every query unnecessarily |

---

## Appendix: Pitching This Project

| Job Requirement | Where in This Project |
|---|---|
| RAG pipeline experience | `pipeline/` — full stack with ablation eval numbers |
| LangChain / LangGraph | `agent/` — StateGraph, typed state, conditional routing |
| Multi-LLM / provider flexibility | `pipeline/llm/` — Factory + identical ABC |
| Production-grade | Celery workers, PostgreSQL, Langfuse, Docker Compose |
| Multilingual NLP | Language detection at parse time, BGE-M3 in roadmap |
| Prompt engineering | `pipeline/prompts.py` + `agent/prompts.py` — versioned with rationale, structured output |
| Evaluation / metrics | `eval/` — RAGAS + FinanceBench + committed JSON |
| Document AI / OCR | `multimodal/` + `parsers/` — Docling, GPT-4o Vision |
| Azure / cloud | `azure_di_parser.py` stub + Azure certs on CV |
| Agent / automation | LangGraph StateGraph with adaptive retrieval loop |

**One-line pitch:**
> "I built this same architecture in production at Dai-ichi Life processing thousands of insurance documents daily. This public repo is a clean documented version — here are the eval numbers."
