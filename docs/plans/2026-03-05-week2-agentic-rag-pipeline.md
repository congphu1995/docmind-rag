# Week 2 — Agentic RAG Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ask a question → streamed answer with citations and agent trace. `POST /api/v1/chat/` works end-to-end with SSE streaming, conditional HyDE, adaptive retrieval with retry, and parent-chunk expansion.

**Architecture:** LangGraph StateGraph wires agent nodes: `query_analyzer → router → [decomposer] → query_rewriter → retriever → reranker → generator`. Every LLM call goes through `BaseLLMClient` ABC — Claude and GPT-4o switchable per query via `LLMFactory`. SSE streams META event (sources + agent trace), then answer tokens, then `[DONE]`. Retriever searches Qdrant child chunks, fetches parent chunks from PostgreSQL for richer LLM context.

**Tech Stack:** FastAPI, LangGraph, Anthropic SDK, OpenAI SDK, sse-starlette, Qdrant, PostgreSQL (async SQLAlchemy)

> **Depends on Week 1:** All pipeline ABCs, IngestionService, QdrantWrapper, OpenAIEmbedder, PostgreSQL models, core config/logging/exceptions.

> **Week 2 Refactor:** `ContextEnricher` and `MetadataExtractor` currently call `AsyncOpenAI` directly. Task 6 refactors them to accept `BaseLLMClient` via constructor injection.

---

## Task 1: Add Week 2 Dependencies

**Files:**
- Modify: `backend/pyproject.toml`

**Step 1: Add dependencies**

```bash
uv add langgraph anthropic sse-starlette
```

**Step 2: Verify imports**

```bash
uv run python -c "
import langgraph; print(f'langgraph {langgraph.__version__}')
import anthropic; print(f'anthropic {anthropic.__version__}')
import sse_starlette; print('sse_starlette OK')
"
```

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add langgraph, anthropic, sse-starlette for Week 2"
```

---

## Task 2: BaseLLMClient ABC

**Files:**
- Create: `backend/app/pipeline/base/llm_client.py`

**Step 1: Implement**

```python
from abc import ABC, abstractmethod
from typing import AsyncGenerator

from pydantic import BaseModel


class BaseLLMClient(ABC):
    """
    Abstract base — all LLM providers implement this interface.
    Services import BaseLLMClient only, never a concrete class.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        **kwargs,
    ) -> str:
        """Single-turn completion. Returns full text response."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Streaming completion. Yields text chunks."""
        ...

    async def complete_structured(
        self,
        messages: list[dict],
        response_model: type[BaseModel],
        system: str | None = None,
        **kwargs,
    ) -> BaseModel:
        """
        Structured output — returns validated Pydantic model.
        Default: parse JSON from text. OpenAIClient overrides with .parse().
        """
        text = await self.complete(messages, system, **kwargs)
        return response_model.model_validate_json(text)

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
```

**Step 2: Verify import**

```bash
uv run python -c "from backend.app.pipeline.base.llm_client import BaseLLMClient; print('OK')"
```

**Step 3: Commit**

```bash
git add backend/app/pipeline/base/llm_client.py
git commit -m "feat: BaseLLMClient ABC with complete, stream, complete_structured"
```

---

## Task 3: OpenAIClient Implementation

**Files:**
- Create: `backend/app/pipeline/llm/openai_client.py`

**Step 1: Implement**

```python
from typing import AsyncGenerator

from openai import AsyncOpenAI
from pydantic import BaseModel

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.pipeline.base.llm_client import BaseLLMClient


class OpenAIClient(BaseLLMClient):

    def __init__(self, model: str = "gpt-4o"):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        **kwargs,
    ) -> str:
        msgs = self._build_messages(messages, system)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=msgs,
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 4096),
        )
        return response.choices[0].message.content

    async def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        msgs = self._build_messages(messages, system)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=msgs,
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 4096),
            stream=True,
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def complete_structured(
        self,
        messages: list[dict],
        response_model: type[BaseModel],
        system: str | None = None,
        **kwargs,
    ) -> BaseModel:
        """OpenAI structured output — uses .parse() for reliable JSON."""
        msgs = self._build_messages(messages, system)
        response = await self._client.beta.chat.completions.parse(
            model=self._model,
            messages=msgs,
            temperature=kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", 200),
            response_format=response_model,
        )
        return response.choices[0].message.parsed

    def _build_messages(
        self, messages: list[dict], system: str | None
    ) -> list[dict]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        return msgs
```

**Step 2: Commit**

```bash
git add backend/app/pipeline/llm/openai_client.py
git commit -m "feat: OpenAIClient — complete, stream, structured output"
```

---

## Task 4: ClaudeClient Implementation

**Files:**
- Create: `backend/app/pipeline/llm/claude_client.py`

**Step 1: Implement**

```python
from typing import AsyncGenerator

from anthropic import AsyncAnthropic

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.pipeline.base.llm_client import BaseLLMClient


class ClaudeClient(BaseLLMClient):

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        **kwargs,
    ) -> str:
        response = await self._client.messages.create(
            model=self._model,
            system=system or "",
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.1),
        )
        return response.content[0].text

    async def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        async with self._client.messages.stream(
            model=self._model,
            system=system or "",
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.1),
        ) as stream:
            async for text in stream.text_stream:
                yield text
```

**Step 2: Commit**

```bash
git add backend/app/pipeline/llm/claude_client.py
git commit -m "feat: ClaudeClient — complete, stream via Anthropic SDK"
```

---

## Task 5: LLMFactory + Unit Tests

**Files:**
- Create: `backend/app/pipeline/llm/factory.py`
- Create: `tests/unit/pipeline/test_llm.py`

**Step 1: Write tests**

```python
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from backend.app.pipeline.llm.factory import LLMFactory


def test_factory_creates_openai():
    with patch("backend.app.pipeline.llm.openai_client.AsyncOpenAI"):
        client = LLMFactory.create("openai")
        from backend.app.pipeline.llm.openai_client import OpenAIClient
        assert isinstance(client, OpenAIClient)


def test_factory_creates_claude():
    with patch("backend.app.pipeline.llm.claude_client.AsyncAnthropic"):
        client = LLMFactory.create("claude")
        from backend.app.pipeline.llm.claude_client import ClaudeClient
        assert isinstance(client, ClaudeClient)


def test_factory_creates_mini():
    with patch("backend.app.pipeline.llm.openai_client.AsyncOpenAI"):
        client = LLMFactory.create_mini()
        assert client.model_name == "gpt-4o-mini"


def test_factory_raises_on_unknown():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        LLMFactory.create("nonexistent")


def test_factory_default_models():
    with patch("backend.app.pipeline.llm.openai_client.AsyncOpenAI"):
        client = LLMFactory.create("openai")
        assert client.model_name == "gpt-4o"

    with patch("backend.app.pipeline.llm.claude_client.AsyncAnthropic"):
        client = LLMFactory.create("claude")
        assert "claude" in client.model_name


async def test_openai_complete():
    with patch("backend.app.pipeline.llm.openai_client.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        from backend.app.pipeline.llm.openai_client import OpenAIClient
        client = OpenAIClient()
        result = await client.complete([{"role": "user", "content": "Hi"}])
        assert result == "Hello!"


async def test_claude_complete():
    with patch("backend.app.pipeline.llm.claude_client.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello from Claude!")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        from backend.app.pipeline.llm.claude_client import ClaudeClient
        client = ClaudeClient()
        result = await client.complete([{"role": "user", "content": "Hi"}])
        assert result == "Hello from Claude!"
```

**Step 2: Implement `backend/app/pipeline/llm/factory.py`**

```python
from backend.app.core.config import settings
from backend.app.pipeline.base.llm_client import BaseLLMClient


class LLMFactory:

    @staticmethod
    def create(provider: str | None = None, model: str | None = None) -> BaseLLMClient:
        """Create LLM client by provider name."""
        provider = provider or settings.default_llm

        if provider == "openai":
            from backend.app.pipeline.llm.openai_client import OpenAIClient
            return OpenAIClient(model=model or "gpt-4o")
        elif provider == "claude":
            from backend.app.pipeline.llm.claude_client import ClaudeClient
            return ClaudeClient(model=model or "claude-sonnet-4-20250514")

        raise ValueError(
            f"Unknown LLM provider: {provider}. Choose: openai, claude"
        )

    @staticmethod
    def create_mini() -> BaseLLMClient:
        """GPT-4o-mini for HyDE, metadata extraction, enrichment."""
        from backend.app.pipeline.llm.openai_client import OpenAIClient
        return OpenAIClient(model="gpt-4o-mini")
```

**Step 3: Run tests**

```bash
uv run pytest tests/unit/pipeline/test_llm.py -v
```

**Step 4: Commit**

```bash
git add backend/app/pipeline/llm/factory.py tests/unit/pipeline/test_llm.py
git commit -m "feat: LLMFactory + OpenAI/Claude unit tests"
```

---

## Task 6: Refactor ContextEnricher + MetadataExtractor

**Files:**
- Modify: `backend/app/pipeline/chunkers/enricher.py`
- Modify: `backend/app/pipeline/parsers/metadata_extractor.py`
- Modify: `backend/app/services/ingestion.py`

**Step 1: Refactor `backend/app/pipeline/chunkers/enricher.py`**

Replace direct `AsyncOpenAI` with `BaseLLMClient` injection.

```python
"""
Prepend situating context to each child chunk before embedding.
Anthropic technique — +15-20% retrieval precision at one-time index cost.
"""
import asyncio

from backend.app.core.logging import logger
from backend.app.pipeline.base.chunker import Chunk
from backend.app.pipeline.base.llm_client import BaseLLMClient
from backend.app.pipeline.prompts import ENRICHMENT_PROMPT


class ContextEnricher:

    def __init__(self, llm: BaseLLMClient):
        self._llm = llm

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

            context_sentence = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0,
            )

            enriched_content = f"{context_sentence.strip()}\n\n{chunk.content_raw}"

            return Chunk(
                **{k: v for k, v in chunk.__dict__.items() if k != "content"},
                content=enriched_content,
            )
```

**Step 2: Refactor `backend/app/pipeline/parsers/metadata_extractor.py`**

Replace direct `AsyncOpenAI` with `BaseLLMClient` injection.

```python
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
```

**Step 3: Update `backend/app/services/ingestion.py`**

Update `IngestionService.__init__` to inject LLM clients:

Replace these lines in `__init__`:
```python
self._metadata_extractor = MetadataExtractor()
...
self._enricher = ContextEnricher()
```

With:
```python
from backend.app.pipeline.llm.factory import LLMFactory
mini_llm = LLMFactory.create_mini()
self._metadata_extractor = MetadataExtractor(llm=mini_llm)
...
self._enricher = ContextEnricher(llm=mini_llm)
```

Full updated imports and `__init__`:

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
from backend.app.pipeline.llm.factory import LLMFactory
from backend.app.pipeline.parsers.factory import ParserFactory
from backend.app.pipeline.parsers.metadata_extractor import MetadataExtractor
from backend.app.pipeline.parsers.preprocessor import PDFPreprocessor
from backend.app.vectorstore.qdrant_client import QdrantWrapper


class IngestionService:

    def __init__(self):
        mini_llm = LLMFactory.create_mini()
        self._preprocessor = PDFPreprocessor()
        self._metadata_extractor = MetadataExtractor(llm=mini_llm)
        self._router = SmartRouter()
        self._enricher = ContextEnricher(llm=mini_llm)
        self._quality_filter = QualityFilter()
        self._embedder = OpenAIEmbedder()
        self._qdrant = QdrantWrapper()

    # ... rest of ingest() and other methods remain unchanged
```

**Step 4: Run existing tests to confirm no regressions**

```bash
uv run pytest tests/unit/ -v
```

**Step 5: Commit**

```bash
git add backend/app/pipeline/chunkers/enricher.py backend/app/pipeline/parsers/metadata_extractor.py backend/app/services/ingestion.py
git commit -m "refactor: inject BaseLLMClient into ContextEnricher and MetadataExtractor"
```

---

## Task 7: Chat + Query Analysis Schemas

**Files:**
- Create: `backend/app/schemas/chat.py`
- Modify: `backend/app/schemas/pipeline.py`

**Step 1: Create `backend/app/schemas/chat.py`**

```python
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    llm: str = Field(default="openai", pattern="^(claude|openai)$")
    doc_ids: list[str] = []
    history: list[dict] = []
    stream: bool = True


class ChatSource(BaseModel):
    doc_name: str
    page: int
    section: str
    content_preview: str
    score: float
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
    llm_used: str
    hyde_used: bool
    query_type: str
    agent_trace: list[str]


class QueryAnalysis(BaseModel):
    """Structured output for query classification."""
    query_type: str = Field(
        description="One of: factual|analytical|tabular|multi_hop|general|greeting"
    )
    language: str = Field(description="ISO 639-1 code of query language")
    sub_questions: list[str] = Field(
        default_factory=list,
        description="Sub-questions for multi_hop queries, empty otherwise",
    )
    filters: dict = Field(
        default_factory=dict,
        description="Extracted metadata filters: doc_type, date_range, etc.",
    )
```

**Step 2: Commit**

```bash
git add backend/app/schemas/chat.py
git commit -m "feat: chat request/response schemas + QueryAnalysis structured output"
```

---

## Task 8: Agent Prompts

**Files:**
- Create: `backend/app/agent/prompts.py`

**Step 1: Implement**

All query pipeline prompts in one file — versioned, with rationale comments.

```python
"""
All prompts for the agentic query pipeline.
Never put prompts inline in node logic.
Versioned with rationale — see comments.
"""

# ── Query Analysis ─────────────────────────────────────────────
# Used by: query_analyzer node
# Model: gpt-4o-mini (structured output)
# Rationale: One fast LLM call classifies the query so downstream
# nodes can make conditional decisions (HyDE, decomposition, routing).

QUERY_ANALYSIS_SYSTEM = """\
You are a query classifier for a document retrieval system.
Classify the user's query and extract metadata."""

QUERY_ANALYSIS_PROMPT = """\
Analyze this query and classify it.

Query: {query}

Query types:
- factual: asks for a specific fact, number, date, or definition
- analytical: asks for comparison, analysis, reasoning, or explanation
- tabular: asks about data in tables, charts, or structured numbers
- multi_hop: requires combining information from multiple sections or documents
- general: general knowledge question not about any uploaded document
- greeting: hello, hi, thanks, etc.

Respond with the classification."""


# ── Query Rewrite ──────────────────────────────────────────────
# Used by: query_rewriter node
# Model: gpt-4o-mini
# Rationale: Expand abbreviations, resolve references from history,
# make the query self-contained for embedding search.

QUERY_REWRITE_PROMPT = """\
Rewrite this query for optimal document retrieval search.
Expand abbreviations, resolve pronouns, make it self-contained.
Output ONLY the rewritten query — no preamble.

Original query: {query}
Conversation context: {context}"""


# ── HyDE (Hypothetical Document Embedding) ────────────────────
# Used by: query_rewriter node (conditional — see section 3.6)
# Model: gpt-4o-mini, max_tokens=150
# Rationale: Generate a hypothetical answer to improve embedding
# similarity. Only for analytical/multi_hop/vague queries.
# Quality doesn't matter — just needs to land near the right
# embedding neighborhood. 150 tokens keeps latency ~300ms.

HYDE_PROMPT = """\
Write a short passage that would answer this question,
as if it appeared in a professional document.
Be specific with terminology. ~100 words.

Question: {query}"""


# ── Decomposer ─────────────────────────────────────────────────
# Used by: decomposer node (multi_hop only)
# Model: gpt-4o-mini
# Rationale: Break complex multi-hop questions into 2-3 simpler
# sub-questions that can each be answered independently.

DECOMPOSE_PROMPT = """\
Break this complex question into 2-3 simpler sub-questions that \
can each be answered from a single document section.
Output one sub-question per line. No numbering or prefixes.

Complex question: {query}"""


# ── Generation ─────────────────────────────────────────────────
# Used by: generator node
# Model: claude-sonnet-4-20250514 or gpt-4o (user choice)
# Rationale: Main answer generation with citations. System prompt
# sets behavior, user prompt provides context + question.

GENERATION_SYSTEM = """\
You are DocMind, an expert document assistant.
Answer the user's question based ONLY on the provided context.
If the context doesn't contain enough information, say so clearly.

Rules:
- Cite sources using [Source N] format after each claim
- Be precise and specific — use exact numbers, dates, names from context
- For tables, reference specific rows/columns
- If multiple sources conflict, note the discrepancy
- Never fabricate information not in the context"""

GENERATION_PROMPT = """\
Context from retrieved documents:

{context}

---

Question: {query}

Provide a thorough answer with [Source N] citations."""


# ── Direct LLM ────────────────────────────────────────────────
# Used by: direct_llm node (general knowledge, no retrieval)
# Rationale: For questions not about uploaded documents.

DIRECT_LLM_SYSTEM = """\
You are DocMind, a helpful document assistant.
The user asked a general question (not about any uploaded document).
Answer helpfully but briefly. Mention that you can help with \
document-specific questions if they upload files."""


# ── Direct Response ────────────────────────────────────────────
# Used by: direct_response node (greetings)
# Not an LLM call — just a template.

GREETING_RESPONSE = (
    "Hello! I'm DocMind, your document intelligence assistant. "
    "I can help you analyze and answer questions about your uploaded documents. "
    "Upload a PDF, DOCX, or text file to get started, or ask me a question "
    "about documents you've already uploaded."
)
```

**Step 2: Commit**

```bash
git add backend/app/agent/prompts.py
git commit -m "feat: agent prompts — analysis, rewrite, HyDE, generation, decompose"
```

---

## Task 9: RAGAgentState

**Files:**
- Create: `backend/app/agent/state.py`

**Step 1: Implement**

```python
"""
LangGraph typed state — flows through every agent node.
Uses Annotated with operator.add for agent_trace accumulation.
"""
import operator
from typing import Annotated, TypedDict


class RAGAgentState(TypedDict):
    # ── Input ────────────────────────────────────────────────
    original_query: str
    doc_ids: list[str]
    llm_preference: str              # "claude" | "openai"

    # ── Query Understanding ──────────────────────────────────
    query_type: str                  # factual|analytical|tabular|multi_hop|general|greeting
    sub_questions: list[str]
    extracted_filters: dict
    detected_language: str

    # ── Query Rewriting ──────────────────────────────────────
    rewritten_query: str
    hyde_query: str                  # empty string if HyDE skipped
    hyde_used: bool

    # ── Retrieval ────────────────────────────────────────────
    retrieved_chunks: list[dict]
    reranked_chunks: list[dict]
    retrieval_attempts: int
    retrieval_quality: float

    # ── Generation ───────────────────────────────────────────
    answer: str
    citations: list[dict]

    # ── Observability ────────────────────────────────────────
    agent_trace: Annotated[list[str], operator.add]   # accumulates across nodes
    error: str
```

**Step 2: Verify import**

```bash
uv run python -c "from backend.app.agent.state import RAGAgentState; print('OK')"
```

**Step 3: Commit**

```bash
git add backend/app/agent/state.py
git commit -m "feat: RAGAgentState TypedDict with trace accumulation"
```

---

## Task 10: QueryAnalyzerNode + Test

**Files:**
- Create: `backend/app/agent/nodes/query_analyzer.py`
- Create: `tests/unit/agent/test_query_analyzer.py`

**Step 1: Write test**

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from backend.app.agent.nodes.query_analyzer import query_analyzer


def _make_state(query: str) -> dict:
    return {
        "original_query": query,
        "doc_ids": [],
        "llm_preference": "openai",
        "query_type": "",
        "sub_questions": [],
        "extracted_filters": {},
        "detected_language": "",
        "rewritten_query": "",
        "hyde_query": "",
        "hyde_used": False,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "retrieval_attempts": 0,
        "retrieval_quality": 0.0,
        "answer": "",
        "citations": [],
        "agent_trace": [],
        "error": "",
    }


@patch("backend.app.agent.nodes.query_analyzer.LLMFactory")
async def test_factual_query_classified(mock_factory):
    mock_llm = AsyncMock()
    mock_factory.create_mini.return_value = mock_llm

    from backend.app.schemas.chat import QueryAnalysis
    mock_llm.complete_structured = AsyncMock(
        return_value=QueryAnalysis(
            query_type="factual",
            language="en",
            sub_questions=[],
            filters={},
        )
    )

    state = _make_state("What is the revenue in Q1 2024?")
    result = await query_analyzer(state)

    assert result["query_type"] == "factual"
    assert result["detected_language"] == "en"
    assert len(result["agent_trace"]) > 0


@patch("backend.app.agent.nodes.query_analyzer.LLMFactory")
async def test_greeting_classified(mock_factory):
    mock_llm = AsyncMock()
    mock_factory.create_mini.return_value = mock_llm

    from backend.app.schemas.chat import QueryAnalysis
    mock_llm.complete_structured = AsyncMock(
        return_value=QueryAnalysis(
            query_type="greeting",
            language="en",
        )
    )

    state = _make_state("Hello!")
    result = await query_analyzer(state)

    assert result["query_type"] == "greeting"


@patch("backend.app.agent.nodes.query_analyzer.LLMFactory")
async def test_analysis_failure_defaults_to_factual(mock_factory):
    mock_llm = AsyncMock()
    mock_factory.create_mini.return_value = mock_llm
    mock_llm.complete_structured = AsyncMock(side_effect=Exception("LLM down"))

    state = _make_state("What is the policy limit?")
    result = await query_analyzer(state)

    assert result["query_type"] == "factual"
    assert result["detected_language"] == "en"
```

**Step 2: Implement `backend/app/agent/nodes/query_analyzer.py`**

```python
"""
First node in the agent pipeline.
Classifies query type, detects language, extracts metadata filters.
"""
from backend.app.agent.prompts import QUERY_ANALYSIS_PROMPT, QUERY_ANALYSIS_SYSTEM
from backend.app.agent.state import RAGAgentState
from backend.app.core.logging import logger
from backend.app.pipeline.llm.factory import LLMFactory
from backend.app.schemas.chat import QueryAnalysis


async def query_analyzer(state: RAGAgentState) -> dict:
    query = state["original_query"]
    log = logger.bind(node="query_analyzer")

    try:
        llm = LLMFactory.create_mini()
        analysis = await llm.complete_structured(
            messages=[
                {
                    "role": "user",
                    "content": QUERY_ANALYSIS_PROMPT.format(query=query),
                }
            ],
            response_model=QueryAnalysis,
            system=QUERY_ANALYSIS_SYSTEM,
            max_tokens=200,
            temperature=0,
        )

        log.info(
            "query_analyzed",
            query_type=analysis.query_type,
            language=analysis.language,
        )

        return {
            "query_type": analysis.query_type,
            "detected_language": analysis.language,
            "sub_questions": analysis.sub_questions,
            "extracted_filters": analysis.filters,
            "agent_trace": [
                f"Query classified as: {analysis.query_type} "
                f"(lang={analysis.language})"
            ],
        }

    except Exception as e:
        log.warning("query_analysis_failed", error=str(e))
        return {
            "query_type": "factual",
            "detected_language": "en",
            "sub_questions": [],
            "extracted_filters": {},
            "agent_trace": [f"Query analysis failed, defaulting to factual: {e}"],
        }
```

**Step 3: Run tests**

```bash
uv run pytest tests/unit/agent/test_query_analyzer.py -v
```

**Step 4: Commit**

```bash
git add backend/app/agent/nodes/query_analyzer.py tests/unit/agent/test_query_analyzer.py
git commit -m "feat: QueryAnalyzerNode — classifies query type + language"
```

---

## Task 11: QueryRewriterNode with Conditional HyDE + Test

**Files:**
- Create: `backend/app/agent/nodes/query_rewriter.py`
- Create: `tests/unit/agent/test_query_rewriter.py`

**Step 1: Write test**

```python
import pytest
from unittest.mock import AsyncMock, patch

from backend.app.agent.nodes.query_rewriter import query_rewriter, _should_use_hyde


def _make_state(query: str, query_type: str = "factual") -> dict:
    return {
        "original_query": query,
        "doc_ids": [],
        "llm_preference": "openai",
        "query_type": query_type,
        "sub_questions": [],
        "extracted_filters": {},
        "detected_language": "en",
        "rewritten_query": "",
        "hyde_query": "",
        "hyde_used": False,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "retrieval_attempts": 0,
        "retrieval_quality": 0.0,
        "answer": "",
        "citations": [],
        "agent_trace": [],
        "error": "",
    }


def test_hyde_skipped_for_greeting():
    assert _should_use_hyde("Hello!", "greeting") is False


def test_hyde_skipped_for_general():
    assert _should_use_hyde("What is the weather?", "general") is False


def test_hyde_skipped_for_factual():
    assert _should_use_hyde("What is the revenue?", "factual") is False


def test_hyde_used_for_analytical():
    assert _should_use_hyde("How does revenue compare?", "analytical") is True


def test_hyde_used_for_multi_hop():
    assert _should_use_hyde("Impact of policy on claims?", "multi_hop") is True


@patch("backend.app.agent.nodes.query_rewriter.LLMFactory")
async def test_rewriter_with_hyde(mock_factory):
    mock_llm = AsyncMock()
    mock_factory.create_mini.return_value = mock_llm
    mock_llm.complete = AsyncMock(
        side_effect=["expanded query about revenue comparison", "hypothetical answer about revenue"]
    )

    state = _make_state(
        "How does the revenue compare between Q1 and Q2?", "analytical"
    )
    result = await query_rewriter(state)

    assert result["hyde_used"] is True
    assert result["hyde_query"] != ""
    assert result["rewritten_query"] != ""


@patch("backend.app.agent.nodes.query_rewriter.LLMFactory")
async def test_rewriter_without_hyde(mock_factory):
    mock_llm = AsyncMock()
    mock_factory.create_mini.return_value = mock_llm
    mock_llm.complete = AsyncMock(return_value="revenue Q1")

    state = _make_state("Revenue Q1", "factual")
    result = await query_rewriter(state)

    assert result["hyde_used"] is False
    assert result["hyde_query"] == ""
```

**Step 2: Implement `backend/app/agent/nodes/query_rewriter.py`**

```python
"""
Rewrites query for optimal retrieval.
Conditional HyDE: only for analytical and multi_hop queries.
Decision based on query_analyzer LLM classification.
"""
import time

from backend.app.agent.prompts import HYDE_PROMPT, QUERY_REWRITE_PROMPT
from backend.app.agent.state import RAGAgentState
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.pipeline.llm.factory import LLMFactory


def _should_use_hyde(query: str, query_type: str) -> bool:
    """
    Decide whether HyDE is beneficial for this query.
    Relies on query_analyzer LLM classification — no manual regex rules.
    """
    if query_type in ("greeting", "general"):
        return False
    if query_type in ("multi_hop", "analytical"):
        return True
    return False


async def query_rewriter(state: RAGAgentState) -> dict:
    query = state["original_query"]
    query_type = state["query_type"]
    log = logger.bind(node="query_rewriter")

    llm = LLMFactory.create_mini()

    # Rewrite: expand abbreviations, resolve references
    rewritten = await llm.complete(
        messages=[
            {
                "role": "user",
                "content": QUERY_REWRITE_PROMPT.format(
                    query=query,
                    context="",  # TODO: pass conversation history
                ),
            }
        ],
        max_tokens=150,
        temperature=0,
    )
    rewritten = rewritten.strip()

    # Conditional HyDE
    hyde_query = ""
    hyde_used = False
    trace_parts = [f"Rewritten: '{rewritten}'"]

    if _should_use_hyde(query, query_type):
        start = time.perf_counter()
        hyde_query = await llm.complete(
            messages=[
                {"role": "user", "content": HYDE_PROMPT.format(query=rewritten)}
            ],
            max_tokens=settings.hyde_max_tokens,
            temperature=0.3,
        )
        hyde_query = hyde_query.strip()
        hyde_used = True
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        trace_parts.append(
            f"HyDE: used — {query_type} query, "
            f"generated {len(hyde_query.split())} words in {elapsed_ms}ms"
        )
        log.info("hyde_generated", query_type=query_type, elapsed_ms=elapsed_ms)
    else:
        reason = _hyde_skip_reason(query, query_type)
        trace_parts.append(f"HyDE: skipped — {reason}")

    return {
        "rewritten_query": rewritten,
        "hyde_query": hyde_query,
        "hyde_used": hyde_used,
        "agent_trace": ["; ".join(trace_parts)],
    }


def _hyde_skip_reason(query: str, query_type: str) -> str:
    if query_type in ("greeting", "general"):
        return f"{query_type} query"
    return f"{query_type} query — not analytical or multi_hop"
```

**Step 3: Run tests**

```bash
uv run pytest tests/unit/agent/test_query_rewriter.py -v
```

**Step 4: Commit**

```bash
git add backend/app/agent/nodes/query_rewriter.py tests/unit/agent/test_query_rewriter.py
git commit -m "feat: QueryRewriterNode with conditional HyDE"
```

---

## Task 12: DecomposerNode

**Files:**
- Create: `backend/app/agent/nodes/decomposer.py`

**Step 1: Implement**

```python
"""
Decomposes multi_hop questions into 2-3 simpler sub-questions.
Only invoked when query_type == 'multi_hop'.
"""
from backend.app.agent.prompts import DECOMPOSE_PROMPT
from backend.app.agent.state import RAGAgentState
from backend.app.core.logging import logger
from backend.app.pipeline.llm.factory import LLMFactory


async def decomposer(state: RAGAgentState) -> dict:
    query = state["original_query"]
    log = logger.bind(node="decomposer")

    llm = LLMFactory.create_mini()
    response = await llm.complete(
        messages=[
            {"role": "user", "content": DECOMPOSE_PROMPT.format(query=query)}
        ],
        max_tokens=200,
        temperature=0,
    )

    sub_questions = [
        line.strip()
        for line in response.strip().split("\n")
        if line.strip() and len(line.strip()) > 5
    ]

    if not sub_questions:
        sub_questions = [query]

    log.info("decomposed", sub_questions=len(sub_questions))

    return {
        "sub_questions": sub_questions,
        "agent_trace": [
            f"Decomposed into {len(sub_questions)} sub-questions: "
            + "; ".join(sub_questions)
        ],
    }
```

**Step 2: Commit**

```bash
git add backend/app/agent/nodes/decomposer.py
git commit -m "feat: DecomposerNode — breaks multi_hop into sub-questions"
```

---

## Task 13: AdaptiveRetrieverNode + Parent Fetch + Test

**Files:**
- Create: `backend/app/agent/nodes/retriever.py`
- Create: `tests/unit/agent/test_retriever.py`

**Step 1: Write test**

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from backend.app.agent.nodes.retriever import retriever_node, _assess_quality


def _make_state(
    rewritten_query: str = "test query",
    hyde_query: str = "",
    sub_questions: list = None,
) -> dict:
    return {
        "original_query": "test",
        "doc_ids": [],
        "llm_preference": "openai",
        "query_type": "factual",
        "sub_questions": sub_questions or [],
        "extracted_filters": {},
        "detected_language": "en",
        "rewritten_query": rewritten_query,
        "hyde_query": hyde_query,
        "hyde_used": bool(hyde_query),
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "retrieval_attempts": 0,
        "retrieval_quality": 0.0,
        "answer": "",
        "citations": [],
        "agent_trace": [],
        "error": "",
    }


def test_assess_quality_high_scores():
    results = [MagicMock(score=0.9), MagicMock(score=0.85), MagicMock(score=0.8)]
    assert _assess_quality(results) > 0.8


def test_assess_quality_low_scores():
    results = [MagicMock(score=0.3), MagicMock(score=0.2)]
    assert _assess_quality(results) < 0.4


def test_assess_quality_empty():
    assert _assess_quality([]) == 0.0


@patch("backend.app.agent.nodes.retriever._fetch_parents")
@patch("backend.app.agent.nodes.retriever.QdrantWrapper")
@patch("backend.app.agent.nodes.retriever.OpenAIEmbedder")
async def test_retriever_uses_hyde_query(mock_embedder_cls, mock_qdrant_cls, mock_fetch):
    mock_embedder = AsyncMock()
    mock_embedder_cls.return_value = mock_embedder
    mock_embedder.embed_single = AsyncMock(return_value=[0.1] * 1536)

    mock_qdrant = MagicMock()
    mock_qdrant_cls.return_value = mock_qdrant
    mock_qdrant.search = AsyncMock(return_value=[
        MagicMock(score=0.9, payload={"parent_id": "p1", "doc_id": "d1"})
    ])

    mock_fetch.return_value = [{"content": "result", "score": 0.9, "chunk_id": "c1"}]

    state = _make_state(hyde_query="hypothetical answer about revenue")
    result = await retriever_node(state)

    assert len(result["retrieved_chunks"]) > 0
    assert result["retrieval_attempts"] >= 1


@patch("backend.app.agent.nodes.retriever._fetch_parents")
@patch("backend.app.agent.nodes.retriever.QdrantWrapper")
@patch("backend.app.agent.nodes.retriever.OpenAIEmbedder")
async def test_retriever_retries_on_low_quality(mock_embedder_cls, mock_qdrant_cls, mock_fetch):
    mock_embedder = AsyncMock()
    mock_embedder_cls.return_value = mock_embedder
    mock_embedder.embed_single = AsyncMock(return_value=[0.1] * 1536)

    mock_qdrant = MagicMock()
    mock_qdrant_cls.return_value = mock_qdrant
    # First attempt: low scores, second: high scores
    mock_qdrant.search = AsyncMock(side_effect=[
        [MagicMock(score=0.3, payload={"parent_id": "p1", "doc_id": "d1"})],
        [MagicMock(score=0.85, payload={"parent_id": "p1", "doc_id": "d1"})],
    ])

    mock_fetch.return_value = [{"content": "result", "score": 0.85, "chunk_id": "c1"}]

    with patch("backend.app.agent.nodes.retriever.LLMFactory") as mock_llm_factory:
        mock_llm = AsyncMock()
        mock_llm_factory.create_mini.return_value = mock_llm
        mock_llm.complete = AsyncMock(return_value="expanded query")

        state = _make_state()
        result = await retriever_node(state)

        assert result["retrieval_attempts"] == 2
```

**Step 2: Implement `backend/app/agent/nodes/retriever.py`**

```python
"""
Adaptive retriever with retry loop.
1. Dense search in Qdrant (child chunks)
2. Assess quality score
3. Retry with expanded query if quality < threshold
4. Fetch parent chunks from PostgreSQL for richer LLM context
"""
from sqlalchemy import select

from backend.app.agent.state import RAGAgentState
from backend.app.core.config import settings
from backend.app.core.database import AsyncSessionLocal
from backend.app.core.logging import logger
from backend.app.models.document import ParentChunk
from backend.app.pipeline.embedders.openai_embedder import OpenAIEmbedder
from backend.app.pipeline.llm.factory import LLMFactory
from backend.app.vectorstore.qdrant_client import QdrantWrapper


MAX_ATTEMPTS = 3


def _assess_quality(results: list) -> float:
    """Average score of top-5 results. 0.0 if empty."""
    if not results:
        return 0.0
    scores = [r.score for r in results[:5]]
    return sum(scores) / len(scores)


async def _fetch_parents(child_results: list) -> list[dict]:
    """
    Look up parent chunks from PostgreSQL.
    For each child, return the parent's full content for richer LLM context.
    Atomic chunks (no parent_id) pass through directly.
    """
    parent_ids = {
        r.payload.get("parent_id")
        for r in child_results
        if r.payload.get("parent_id")
    }

    parents = {}
    if parent_ids:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(ParentChunk).where(ParentChunk.chunk_id.in_(parent_ids))
            )
            parents = {p.chunk_id: p for p in result.scalars().all()}

    chunks = []
    seen_parents: set[str] = set()

    for r in child_results:
        parent_id = r.payload.get("parent_id")

        if parent_id and parent_id in parents and parent_id not in seen_parents:
            parent = parents[parent_id]
            chunks.append({
                "content": parent.content_raw,
                "content_markdown": parent.content_markdown,
                "doc_id": parent.doc_id,
                "doc_name": r.payload.get("doc_name", ""),
                "page": parent.page,
                "section": parent.section,
                "type": parent.type,
                "score": r.score,
                "chunk_id": parent.chunk_id,
            })
            seen_parents.add(parent_id)

        elif not parent_id:
            # Atomic chunk (table, figure) — no parent, use child directly
            chunks.append({
                "content": r.payload.get("content_raw", ""),
                "content_markdown": r.payload.get("content_markdown"),
                "doc_id": r.payload.get("doc_id", ""),
                "doc_name": r.payload.get("doc_name", ""),
                "page": r.payload.get("page", 0),
                "section": r.payload.get("section", ""),
                "type": r.payload.get("type", "text"),
                "score": r.score,
                "chunk_id": r.payload.get("chunk_id", ""),
            })

    return chunks


async def retriever_node(state: RAGAgentState) -> dict:
    log = logger.bind(node="retriever")

    embedder = OpenAIEmbedder()
    qdrant = QdrantWrapper()

    # Use HyDE query if available, otherwise rewritten query
    query_text = state.get("hyde_query") or state["rewritten_query"] or state["original_query"]

    # Build Qdrant filters
    filters = {}
    if state.get("doc_ids"):
        filters["doc_ids"] = state["doc_ids"]
    if state.get("detected_language") and state["detected_language"] != "en":
        filters["language"] = state["detected_language"]

    quality = 0.0
    results = []
    attempt = 0

    for attempt in range(MAX_ATTEMPTS):
        vector = await embedder.embed_single(query_text)
        results = await qdrant.search(
            vector=vector,
            top_k=settings.retrieval_top_k,
            filters=filters if filters else None,
        )

        quality = _assess_quality(results)
        log.info(
            "retrieval_attempt",
            attempt=attempt + 1,
            results=len(results),
            quality=round(quality, 3),
        )

        if quality >= settings.retrieval_quality_threshold or attempt == MAX_ATTEMPTS - 1:
            break

        # Retry: expand query
        if attempt == 0:
            llm = LLMFactory.create_mini()
            query_text = await llm.complete(
                messages=[{
                    "role": "user",
                    "content": f"Rephrase this for better document search. "
                               f"Add synonyms and related terms. "
                               f"Output ONLY the query:\n\n{state['original_query']}",
                }],
                max_tokens=100,
                temperature=0.3,
            )
            query_text = query_text.strip()
        elif attempt == 1 and state.get("sub_questions"):
            # Attempt 3: try first sub-question
            query_text = state["sub_questions"][0]

    # Fetch parent chunks from PostgreSQL
    chunks = await _fetch_parents(results)

    log.info(
        "retrieval_done",
        attempts=attempt + 1,
        quality=round(quality, 3),
        chunks=len(chunks),
    )

    return {
        "retrieved_chunks": chunks,
        "retrieval_attempts": attempt + 1,
        "retrieval_quality": quality,
        "agent_trace": [
            f"Retrieved {len(chunks)} chunks in {attempt + 1} attempt(s) "
            f"(quality={quality:.2f})"
        ],
    }
```

**Step 3: Run tests**

```bash
uv run pytest tests/unit/agent/test_retriever.py -v
```

**Step 4: Commit**

```bash
git add backend/app/agent/nodes/retriever.py tests/unit/agent/test_retriever.py
git commit -m "feat: AdaptiveRetrieverNode with retry loop + parent fetch"
```

---

## Task 14: IdentityReranker + RerankerNode

**Files:**
- Create: `backend/app/pipeline/rerankers/identity_reranker.py`
- Create: `backend/app/pipeline/rerankers/factory.py`
- Create: `backend/app/agent/nodes/reranker.py`

**Step 1: Implement `backend/app/pipeline/rerankers/identity_reranker.py`**

```python
from backend.app.pipeline.base.reranker import BaseReranker


class IdentityReranker(BaseReranker):
    """Passthrough reranker — returns chunks unchanged. Default until Cohere is added."""

    async def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_n: int,
    ) -> list[dict]:
        return chunks[:top_n]
```

**Step 2: Implement `backend/app/pipeline/rerankers/factory.py`**

```python
from backend.app.pipeline.base.reranker import BaseReranker
from backend.app.pipeline.rerankers.identity_reranker import IdentityReranker


class RerankerFactory:

    @staticmethod
    def create(strategy: str = "identity") -> BaseReranker:
        rerankers = {
            "identity": IdentityReranker,
        }
        if strategy not in rerankers:
            raise ValueError(
                f"Unknown reranker: {strategy}. Choose: {list(rerankers.keys())}"
            )
        return rerankers[strategy]()
```

**Step 3: Implement `backend/app/agent/nodes/reranker.py`**

```python
"""
Reranks retrieved chunks. Identity (passthrough) by default.
Swap to Cohere via config — no code changes needed.
"""
from backend.app.agent.state import RAGAgentState
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.pipeline.rerankers.factory import RerankerFactory


async def reranker_node(state: RAGAgentState) -> dict:
    log = logger.bind(node="reranker")

    reranker = RerankerFactory.create()
    chunks = state.get("retrieved_chunks", [])

    reranked = await reranker.rerank(
        query=state["original_query"],
        chunks=chunks,
        top_n=settings.retrieval_top_n,
    )

    log.info("reranked", before=len(chunks), after=len(reranked))

    return {
        "reranked_chunks": reranked,
        "agent_trace": [
            f"Reranked: {len(chunks)} → {len(reranked)} chunks "
            f"(strategy=identity)"
        ],
    }
```

**Step 4: Commit**

```bash
git add backend/app/pipeline/rerankers/ backend/app/agent/nodes/reranker.py
git commit -m "feat: IdentityReranker + RerankerFactory + RerankerNode"
```

---

## Task 15: GeneratorNode + DirectResponse + DirectLLM

**Files:**
- Create: `backend/app/agent/nodes/generator.py`
- Create: `tests/unit/agent/test_generator.py`

**Step 1: Write test**

```python
import pytest
from unittest.mock import AsyncMock, patch

from backend.app.agent.nodes.generator import (
    generator_node,
    direct_response,
    direct_llm,
    _build_context,
    _extract_citations,
)


def _make_state(chunks: list = None, llm_preference: str = "openai") -> dict:
    return {
        "original_query": "What is the revenue?",
        "doc_ids": [],
        "llm_preference": llm_preference,
        "query_type": "factual",
        "sub_questions": [],
        "extracted_filters": {},
        "detected_language": "en",
        "rewritten_query": "What is the revenue?",
        "hyde_query": "",
        "hyde_used": False,
        "retrieved_chunks": [],
        "reranked_chunks": chunks or [],
        "retrieval_attempts": 1,
        "retrieval_quality": 0.8,
        "answer": "",
        "citations": [],
        "agent_trace": [],
        "error": "",
    }


def test_build_context_formats_sources():
    chunks = [
        {"content": "Revenue was $10M", "doc_name": "report.pdf", "page": 5, "section": "Financials"},
        {"content": "Q2 grew 15%", "doc_name": "report.pdf", "page": 8, "section": "Growth"},
    ]
    context = _build_context(chunks)
    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "Revenue was $10M" in context


def test_build_context_empty():
    assert _build_context([]) == "No relevant context found."


def test_extract_citations():
    answer = "Revenue was $10M [Source 1]. Growth was 15% [Source 2]."
    chunks = [
        {"doc_name": "a.pdf", "page": 1, "section": "A", "content": "Rev", "score": 0.9, "chunk_id": "c1"},
        {"doc_name": "b.pdf", "page": 2, "section": "B", "content": "Growth", "score": 0.8, "chunk_id": "c2"},
    ]
    citations = _extract_citations(answer, chunks)
    assert len(citations) == 2
    assert citations[0]["doc_name"] == "a.pdf"


async def test_direct_response_returns_greeting():
    state = _make_state()
    state["query_type"] = "greeting"
    result = await direct_response(state)
    assert result["answer"] != ""
    assert result["citations"] == []


@patch("backend.app.agent.nodes.generator.LLMFactory")
async def test_direct_llm_no_retrieval(mock_factory):
    mock_llm = AsyncMock()
    mock_factory.create.return_value = mock_llm
    mock_llm.complete = AsyncMock(return_value="The sky is blue.")

    state = _make_state()
    state["query_type"] = "general"
    result = await direct_llm(state)

    assert result["answer"] == "The sky is blue."
    assert result["citations"] == []


@patch("backend.app.agent.nodes.generator.LLMFactory")
async def test_generator_produces_answer(mock_factory):
    mock_llm = AsyncMock()
    mock_factory.create.return_value = mock_llm
    mock_llm.complete = AsyncMock(
        return_value="Revenue was $10M [Source 1]."
    )

    chunks = [
        {"content": "Revenue was $10M", "doc_name": "report.pdf", "page": 5,
         "section": "Financials", "score": 0.9, "chunk_id": "c1"},
    ]
    state = _make_state(chunks=chunks)
    result = await generator_node(state)

    assert "Revenue" in result["answer"]
    assert len(result["citations"]) >= 1
```

**Step 2: Implement `backend/app/agent/nodes/generator.py`**

```python
"""
Final generation node. Builds prompt with enriched context, generates answer
with citations, and provides direct_response/direct_llm for non-retrieval paths.
"""
import re

from backend.app.agent.prompts import (
    DIRECT_LLM_SYSTEM,
    GENERATION_PROMPT,
    GENERATION_SYSTEM,
    GREETING_RESPONSE,
)
from backend.app.agent.state import RAGAgentState
from backend.app.core.logging import logger
from backend.app.pipeline.llm.factory import LLMFactory


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as numbered sources for the LLM."""
    if not chunks:
        return "No relevant context found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        source_label = (
            f"[Source {i}] "
            f"{chunk.get('doc_name', 'Unknown')} — "
            f"Page {chunk.get('page', '?')}, "
            f"Section: {chunk.get('section', 'N/A')}"
        )
        content = chunk.get("content_markdown") or chunk.get("content", "")
        parts.append(f"{source_label}\n{content}")

    return "\n\n---\n\n".join(parts)


def _extract_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """Extract [Source N] references from the answer and map to chunk metadata."""
    pattern = r"\[Source (\d+)\]"
    referenced = set(int(m) for m in re.findall(pattern, answer))

    citations = []
    for i, chunk in enumerate(chunks, 1):
        if i in referenced:
            citations.append({
                "source_num": i,
                "doc_name": chunk.get("doc_name", ""),
                "page": chunk.get("page", 0),
                "section": chunk.get("section", ""),
                "content_preview": chunk.get("content", "")[:200],
                "score": chunk.get("score", 0.0),
                "chunk_id": chunk.get("chunk_id", ""),
            })

    return citations


async def generator_node(state: RAGAgentState) -> dict:
    log = logger.bind(node="generator")

    llm = LLMFactory.create(state["llm_preference"])
    chunks = state.get("reranked_chunks", [])
    context = _build_context(chunks)

    answer = await llm.complete(
        messages=[
            {
                "role": "user",
                "content": GENERATION_PROMPT.format(
                    context=context,
                    query=state["original_query"],
                ),
            }
        ],
        system=GENERATION_SYSTEM,
        max_tokens=4096,
        temperature=0.1,
    )

    citations = _extract_citations(answer, chunks)

    log.info(
        "generation_done",
        llm=llm.model_name,
        citations=len(citations),
        answer_words=len(answer.split()),
    )

    return {
        "answer": answer,
        "citations": citations,
        "agent_trace": [
            f"Generated answer with {len(citations)} citations "
            f"(llm={llm.model_name})"
        ],
    }


async def direct_response(state: RAGAgentState) -> dict:
    """Greeting response — no LLM call needed."""
    return {
        "answer": GREETING_RESPONSE,
        "citations": [],
        "agent_trace": ["Direct response — greeting detected"],
    }


async def direct_llm(state: RAGAgentState) -> dict:
    """General knowledge — LLM without retrieval."""
    llm = LLMFactory.create(state["llm_preference"])

    answer = await llm.complete(
        messages=[
            {"role": "user", "content": state["original_query"]}
        ],
        system=DIRECT_LLM_SYSTEM,
        max_tokens=1024,
    )

    return {
        "answer": answer,
        "citations": [],
        "agent_trace": [
            f"Direct LLM — no retrieval (llm={llm.model_name})"
        ],
    }
```

**Step 3: Run tests**

```bash
uv run pytest tests/unit/agent/test_generator.py -v
```

**Step 4: Commit**

```bash
git add backend/app/agent/nodes/generator.py tests/unit/agent/test_generator.py
git commit -m "feat: GeneratorNode + DirectResponse + DirectLLM with citations"
```

---

## Task 16: LangGraph StateGraph Wiring + Test

**Files:**
- Create: `backend/app/agent/graph.py`
- Create: `tests/unit/agent/test_graph.py`

**Step 1: Write test**

```python
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from backend.app.agent.graph import build_graph


def test_graph_compiles():
    """Graph should compile without errors."""
    graph = build_graph()
    assert graph is not None


def test_graph_has_expected_nodes():
    """Verify all expected nodes are in the graph."""
    graph = build_graph()
    # LangGraph compiled graph nodes are accessible
    node_names = set(graph.get_graph().nodes.keys())
    expected = {
        "query_analyzer",
        "query_rewriter",
        "decomposer",
        "retriever",
        "reranker",
        "generator",
        "direct_response",
        "direct_llm",
    }
    assert expected.issubset(node_names)
```

**Step 2: Implement `backend/app/agent/graph.py`**

```python
"""
LangGraph StateGraph — wires all agent nodes with conditional routing.

Flow:
  query_analyzer → router (conditional)
    ├── factual/analytical/tabular → query_rewriter → retriever → reranker → generator → END
    ├── multi_hop → decomposer → query_rewriter → retriever → reranker → generator → END
    ├── general → direct_llm → END
    └── greeting → direct_response → END
"""
from langgraph.graph import END, StateGraph

from backend.app.agent.nodes.decomposer import decomposer
from backend.app.agent.nodes.generator import direct_llm, direct_response, generator_node
from backend.app.agent.nodes.query_analyzer import query_analyzer
from backend.app.agent.nodes.query_rewriter import query_rewriter
from backend.app.agent.nodes.reranker import reranker_node
from backend.app.agent.nodes.retriever import retriever_node
from backend.app.agent.state import RAGAgentState


def _route_query(state: RAGAgentState) -> str:
    """Conditional routing after query_analyzer."""
    query_type = state.get("query_type", "factual")

    if query_type == "greeting":
        return "direct_response"
    if query_type == "general":
        return "direct_llm"
    if query_type == "multi_hop":
        return "decomposer"

    # factual, analytical, tabular → query_rewriter
    return "query_rewriter"


def build_graph():
    """Build and compile the RAG agent graph."""
    graph = StateGraph(RAGAgentState)

    # Add nodes
    graph.add_node("query_analyzer", query_analyzer)
    graph.add_node("query_rewriter", query_rewriter)
    graph.add_node("decomposer", decomposer)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reranker", reranker_node)
    graph.add_node("generator", generator_node)
    graph.add_node("direct_response", direct_response)
    graph.add_node("direct_llm", direct_llm)

    # Entry point
    graph.set_entry_point("query_analyzer")

    # Conditional routing after analysis
    graph.add_conditional_edges(
        "query_analyzer",
        _route_query,
        {
            "query_rewriter": "query_rewriter",
            "decomposer": "decomposer",
            "direct_response": "direct_response",
            "direct_llm": "direct_llm",
        },
    )

    # Decomposer → query_rewriter
    graph.add_edge("decomposer", "query_rewriter")

    # Main retrieval path
    graph.add_edge("query_rewriter", "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "generator")

    # Terminal edges
    graph.add_edge("generator", END)
    graph.add_edge("direct_response", END)
    graph.add_edge("direct_llm", END)

    return graph.compile()
```

**Step 3: Run tests**

```bash
uv run pytest tests/unit/agent/test_graph.py -v
```

**Step 4: Commit**

```bash
git add backend/app/agent/graph.py tests/unit/agent/test_graph.py
git commit -m "feat: LangGraph StateGraph with conditional routing"
```

---

## Task 17: RAGService

**Files:**
- Create: `backend/app/services/rag.py`

**Step 1: Implement**

```python
"""
Orchestrates the RAG agent pipeline.
- query(): Non-streaming — runs full graph, returns complete response.
- stream_query(): Streaming — runs nodes sequentially, streams generation via SSE.
"""
import json
from typing import AsyncGenerator

from backend.app.agent.graph import build_graph
from backend.app.agent.nodes.generator import (
    _build_context,
    _extract_citations,
    direct_llm,
    direct_response,
)
from backend.app.agent.nodes.query_analyzer import query_analyzer
from backend.app.agent.nodes.query_rewriter import query_rewriter
from backend.app.agent.nodes.decomposer import decomposer
from backend.app.agent.nodes.reranker import reranker_node
from backend.app.agent.nodes.retriever import retriever_node
from backend.app.agent.prompts import GENERATION_PROMPT, GENERATION_SYSTEM
from backend.app.agent.state import RAGAgentState
from backend.app.core.logging import logger
from backend.app.pipeline.llm.factory import LLMFactory
from backend.app.schemas.chat import ChatRequest


class RAGService:

    def __init__(self):
        self._graph = build_graph()

    def _build_initial_state(self, request: ChatRequest) -> RAGAgentState:
        return {
            "original_query": request.question,
            "doc_ids": request.doc_ids,
            "llm_preference": request.llm,
            "query_type": "",
            "sub_questions": [],
            "extracted_filters": {},
            "detected_language": "",
            "rewritten_query": "",
            "hyde_query": "",
            "hyde_used": False,
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "retrieval_attempts": 0,
            "retrieval_quality": 0.0,
            "answer": "",
            "citations": [],
            "agent_trace": [],
            "error": "",
        }

    async def query(self, request: ChatRequest) -> dict:
        """Non-streaming: run full graph, return complete response."""
        log = logger.bind(question=request.question[:80])
        log.info("rag_query_start")

        state = self._build_initial_state(request)
        result = await self._graph.ainvoke(state)

        log.info(
            "rag_query_done",
            query_type=result.get("query_type"),
            citations=len(result.get("citations", [])),
        )

        return {
            "answer": result.get("answer", ""),
            "sources": result.get("citations", []),
            "llm_used": request.llm,
            "hyde_used": result.get("hyde_used", False),
            "query_type": result.get("query_type", ""),
            "agent_trace": result.get("agent_trace", []),
        }

    async def stream_query(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """
        Streaming: run nodes sequentially, then stream generation.

        SSE format:
        1. __META__{...}__META__  (sources, trace, metadata)
        2. token by token         (streamed answer)
        3. [DONE]                 (terminal signal)
        """
        log = logger.bind(question=request.question[:80])
        log.info("rag_stream_start")

        state = self._build_initial_state(request)

        # Phase 1: Query analysis
        analyzer_result = await query_analyzer(state)
        state = {**state, **analyzer_result}

        query_type = state["query_type"]

        # Handle non-retrieval paths
        if query_type == "greeting":
            result = await direct_response(state)
            state = {**state, **result}
            yield self._build_meta_event(state, request.llm)
            yield state["answer"]
            yield "[DONE]"
            return

        if query_type == "general":
            result = await direct_llm(state)
            state = {**state, **result}
            yield self._build_meta_event(state, request.llm)
            yield state["answer"]
            yield "[DONE]"
            return

        # Phase 2: Decompose (multi_hop only)
        if query_type == "multi_hop":
            decompose_result = await decomposer(state)
            state = {**state, **decompose_result}

        # Phase 3: Query rewrite + conditional HyDE
        rewrite_result = await query_rewriter(state)
        state = {**state, **rewrite_result}

        # Phase 4: Retrieval with adaptive retry
        retrieval_result = await retriever_node(state)
        state = {**state, **retrieval_result}

        # Phase 5: Rerank
        rerank_result = await reranker_node(state)
        state = {**state, **rerank_result}

        # Yield META event (sources + trace before generation)
        yield self._build_meta_event(state, request.llm)

        # Phase 6: Stream generation
        llm = LLMFactory.create(request.llm)
        context = _build_context(state.get("reranked_chunks", []))
        prompt = GENERATION_PROMPT.format(
            context=context,
            query=state["original_query"],
        )

        full_answer = ""
        async for token in llm.stream(
            messages=[{"role": "user", "content": prompt}],
            system=GENERATION_SYSTEM,
            max_tokens=4096,
            temperature=0.1,
        ):
            full_answer += token
            yield token

        yield "[DONE]"

        log.info(
            "rag_stream_done",
            query_type=query_type,
            answer_words=len(full_answer.split()),
        )

    def _build_meta_event(self, state: dict, llm_used: str) -> str:
        """Build __META__ SSE event with sources and agent trace."""
        # Build source list from reranked chunks
        sources = []
        for i, chunk in enumerate(state.get("reranked_chunks", []), 1):
            sources.append({
                "source_num": i,
                "doc_name": chunk.get("doc_name", ""),
                "page": chunk.get("page", 0),
                "section": chunk.get("section", ""),
                "content_preview": chunk.get("content", "")[:200],
                "score": chunk.get("score", 0.0),
            })

        meta = {
            "sources": sources,
            "llm_used": llm_used,
            "hyde_used": state.get("hyde_used", False),
            "query_type": state.get("query_type", ""),
            "agent_trace": state.get("agent_trace", []),
        }

        return f"__META__{json.dumps(meta)}__META__"
```

**Step 2: Commit**

```bash
git add backend/app/services/rag.py
git commit -m "feat: RAGService with streaming SSE + non-streaming query"
```

---

## Task 18: Chat API Endpoint (SSE) + main.py Update

**Files:**
- Create: `backend/app/api/chat.py`
- Modify: `backend/app/main.py`

**Step 1: Create `backend/app/api/chat.py`**

```python
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from backend.app.core.logging import logger
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.rag import RAGService

router = APIRouter()

_rag_service = RAGService()


@router.post("/")
async def chat(request: ChatRequest):
    """
    Chat endpoint.
    stream=true (default): SSE stream with META + tokens + [DONE]
    stream=false: JSON response with full answer.
    """
    try:
        if request.stream:
            return EventSourceResponse(
                _stream_response(request),
                media_type="text/event-stream",
            )

        result = await _rag_service.query(request)
        return result

    except Exception as e:
        logger.error("chat_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_response(request: ChatRequest):
    """Yield SSE events from RAGService.stream_query()."""
    async for event in _rag_service.stream_query(request):
        yield {"data": event}
```

**Step 2: Update `backend/app/main.py`**

Add the chat router import and include:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import chat, documents, health
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
app.include_router(
    chat.router, prefix="/api/v1/chat", tags=["chat"]
)
app.include_router(health.router, prefix="/api/v1", tags=["health"])
```

**Step 3: Commit**

```bash
git add backend/app/api/chat.py backend/app/main.py
git commit -m "feat: chat SSE endpoint + main.py router update"
```

---

## Task 19: Comprehensive Agent Unit Tests

**Files:**
- Create: `tests/unit/agent/test_reranker.py`
- Create: `tests/unit/agent/test_decomposer.py`

**Step 1: Create `tests/unit/agent/test_reranker.py`**

```python
import pytest
from unittest.mock import AsyncMock, patch

from backend.app.agent.nodes.reranker import reranker_node


def _make_state(chunks: list = None) -> dict:
    return {
        "original_query": "What is the revenue?",
        "doc_ids": [],
        "llm_preference": "openai",
        "query_type": "factual",
        "sub_questions": [],
        "extracted_filters": {},
        "detected_language": "en",
        "rewritten_query": "",
        "hyde_query": "",
        "hyde_used": False,
        "retrieved_chunks": chunks or [],
        "reranked_chunks": [],
        "retrieval_attempts": 1,
        "retrieval_quality": 0.8,
        "answer": "",
        "citations": [],
        "agent_trace": [],
        "error": "",
    }


async def test_identity_reranker_limits_to_top_n():
    chunks = [{"content": f"chunk {i}", "score": 1 - i * 0.1} for i in range(10)]
    state = _make_state(chunks=chunks)
    result = await reranker_node(state)
    assert len(result["reranked_chunks"]) == 5  # default top_n


async def test_identity_reranker_preserves_order():
    chunks = [
        {"content": "first", "score": 0.9},
        {"content": "second", "score": 0.8},
    ]
    state = _make_state(chunks=chunks)
    result = await reranker_node(state)
    assert result["reranked_chunks"][0]["content"] == "first"


async def test_reranker_handles_empty_chunks():
    state = _make_state(chunks=[])
    result = await reranker_node(state)
    assert result["reranked_chunks"] == []
```

**Step 2: Create `tests/unit/agent/test_decomposer.py`**

```python
import pytest
from unittest.mock import AsyncMock, patch

from backend.app.agent.nodes.decomposer import decomposer


def _make_state(query: str) -> dict:
    return {
        "original_query": query,
        "doc_ids": [],
        "llm_preference": "openai",
        "query_type": "multi_hop",
        "sub_questions": [],
        "extracted_filters": {},
        "detected_language": "en",
        "rewritten_query": "",
        "hyde_query": "",
        "hyde_used": False,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "retrieval_attempts": 0,
        "retrieval_quality": 0.0,
        "answer": "",
        "citations": [],
        "agent_trace": [],
        "error": "",
    }


@patch("backend.app.agent.nodes.decomposer.LLMFactory")
async def test_decomposer_splits_into_sub_questions(mock_factory):
    mock_llm = AsyncMock()
    mock_factory.create_mini.return_value = mock_llm
    mock_llm.complete = AsyncMock(
        return_value="What was the revenue in Q1?\nWhat was the revenue in Q2?\nHow do they compare?"
    )

    state = _make_state("How did Q1 revenue compare to Q2?")
    result = await decomposer(state)

    assert len(result["sub_questions"]) == 3
    assert "revenue" in result["sub_questions"][0].lower()


@patch("backend.app.agent.nodes.decomposer.LLMFactory")
async def test_decomposer_fallback_on_empty(mock_factory):
    mock_llm = AsyncMock()
    mock_factory.create_mini.return_value = mock_llm
    mock_llm.complete = AsyncMock(return_value="")

    state = _make_state("Complex question here")
    result = await decomposer(state)

    assert len(result["sub_questions"]) == 1
    assert result["sub_questions"][0] == "Complex question here"
```

**Step 3: Run all agent tests**

```bash
uv run pytest tests/unit/agent/ -v
```

**Step 4: Commit**

```bash
git add tests/unit/agent/
git commit -m "test: comprehensive agent node tests — reranker, decomposer"
```

---

## Task 20: Integration Test + Final Verification

**Files:**
- Create: `tests/integration/test_chat_pipeline.py`

**Step 1: Create integration test**

```python
"""
Requires: Qdrant + PostgreSQL running (docker compose up qdrant postgres)
Requires: OPENAI_API_KEY set (for embeddings + gpt-4o-mini)
Run with: pytest tests/integration/test_chat_pipeline.py -m integration
"""
import pytest

from backend.app.schemas.chat import ChatRequest


@pytest.mark.integration
async def test_chat_non_streaming(sample_pdf_path):
    """Ingest a document, then query it non-streaming."""
    from backend.app.services.ingestion import IngestionService
    from backend.app.services.rag import RAGService

    # Ingest
    ingestion = IngestionService()
    ingest_result = await ingestion.ingest(sample_pdf_path, "sample.pdf")
    doc_id = ingest_result["doc_id"]

    # Query
    rag = RAGService()
    request = ChatRequest(
        question="What is this document about?",
        llm="openai",  # Use openai for testing (only needs OPENAI_API_KEY)
        doc_ids=[doc_id],
        stream=False,
    )
    result = await rag.query(request)

    assert result["answer"] != ""
    assert result["query_type"] in (
        "factual", "analytical", "general", "multi_hop", "tabular"
    )

    # Cleanup
    await ingestion.delete_document(doc_id)


@pytest.mark.integration
async def test_chat_greeting():
    """Greeting should not trigger retrieval."""
    from backend.app.services.rag import RAGService

    rag = RAGService()
    request = ChatRequest(
        question="Hello!",
        llm="openai",
        stream=False,
    )
    result = await rag.query(request)

    assert result["answer"] != ""
    assert result["query_type"] == "greeting"
    assert result["sources"] == []


@pytest.mark.integration
async def test_chat_streaming():
    """Streaming should yield META + tokens + DONE."""
    from backend.app.services.rag import RAGService

    rag = RAGService()
    request = ChatRequest(
        question="Hello!",
        llm="openai",
    )

    events = []
    async for event in rag.stream_query(request):
        events.append(event)

    assert any("__META__" in e for e in events)
    assert events[-1] == "[DONE]"
```

**Step 2: Run all unit tests**

```bash
uv run pytest tests/unit/ -v
```

Expected: All pass.

**Step 3: Start infrastructure**

```bash
docker compose up -d
```

**Step 4: Start FastAPI (smoke test)**

```bash
OPENAI_API_KEY=test ANTHROPIC_API_KEY=test uv run uvicorn backend.app.main:app --port 8000
```

In another terminal, verify:
```bash
# Health check
curl http://localhost:8000/api/v1/health

# OpenAPI docs accessible
curl -s http://localhost:8000/openapi.json | python3 -c "import sys,json; d=json.load(sys.stdin); print([p for p in d['paths'] if 'chat' in p])"
```

Expected:
- Health: `{"status": "healthy"}`
- Chat endpoint visible in OpenAPI: `/api/v1/chat/`

**Step 5: Run integration tests (optional — requires real API keys)**

```bash
uv run pytest tests/integration/test_chat_pipeline.py -m integration -v
```

**Step 6: Final commit**

```bash
git add -A
git commit -m "feat: Week 2 complete — agentic RAG pipeline with SSE streaming"
```

---

## Summary — What Week 2 Delivers

| Component | Files | Purpose |
|---|---|---|
| **BaseLLMClient** | `pipeline/base/llm_client.py` | ABC for all LLM providers |
| **OpenAIClient** | `pipeline/llm/openai_client.py` | GPT-4o / GPT-4o-mini |
| **ClaudeClient** | `pipeline/llm/claude_client.py` | Claude Sonnet |
| **LLMFactory** | `pipeline/llm/factory.py` | Provider selection by config |
| **Agent Nodes** | `agent/nodes/*.py` | 8 nodes: analyzer, rewriter, decomposer, retriever, reranker, generator, direct_response, direct_llm |
| **StateGraph** | `agent/graph.py` | LangGraph wiring with conditional routing |
| **RAGService** | `services/rag.py` | Orchestration: streaming + non-streaming |
| **Chat API** | `api/chat.py` | SSE endpoint: META + tokens + DONE |
| **Agent Prompts** | `agent/prompts.py` | All prompts versioned with rationale |
| **IdentityReranker** | `pipeline/rerankers/` | Passthrough default, ready for Cohere |
| **Refactored** | `enricher.py`, `metadata_extractor.py` | Now use BaseLLMClient injection |

**End-to-end flow:**
```
POST /api/v1/chat/ { question, llm, doc_ids, stream }
  → query_analyzer (classify + detect language)
  → router (conditional: greeting → direct, general → direct_llm, multi_hop → decomposer)
  → query_rewriter (expand + conditional HyDE)
  → retriever (Qdrant child search → PostgreSQL parent fetch, adaptive retry)
  → reranker (identity passthrough)
  → generator (LLM streaming with [Source N] citations)
  → SSE: __META__ → tokens → [DONE]
```
