# Post-Week 3 Roadmap — Design Document

> 4 features beyond the core 3-week plan, ordered by dependency.

| Field | Value |
|---|---|
| Author | Cong Phu Nguyen |
| Date | 2026-03-08 |
| Status | Approved |
| Scope | CohereReranker, JWT Auth, Prometheus + Grafana, ChunkViewer |

---

## Feature Overview

```
Phase 1: CohereReranker           Retrieval quality improvement
Phase 2: JWT Auth                 User system + API security
Phase 3: Prometheus + Grafana     Operational observability
Phase 4: ChunkViewer page         Chunk inspection UI
```

**Ordering rationale:**
- Cohere first — smallest scope, plugs into existing `BaseReranker` ABC, immediately improves retrieval
- JWT second — once auth exists, all new frontend pages are protected from the start
- Prometheus third — once instrumented, you can measure impact of everything
- ChunkViewer last — needs ingested docs, useful for debugging retrieval quality

---

## Phase 1: CohereReranker

**What it does:** After retrieval returns top-20 chunks from Qdrant, Cohere Rerank v3 re-scores them using a cross-encoder, keeping top-5. Plugs into the existing `BaseReranker` ABC.

**Behavior:**
- `reranker_strategy = "identity"` (default, unchanged) or `"cohere"`
- Model: `rerank-english-v3.0`
- Input: top-20 retrieved chunks + original query
- Output: top-5 re-scored and sorted by relevance
- Fallback: if Cohere API fails, log warning and fall back to identity (return input unchanged)

**Dependency:** `cohere` Python package

**Files:**

| File | Change |
|------|--------|
| `pipeline/rerankers/cohere_reranker.py` | New — `CohereReranker(BaseReranker)` calling `cohere.rerank()` |
| `pipeline/rerankers/factory.py` | Add `"cohere"` case to factory |
| `core/config.py` | Add `reranker_strategy`, `cohere_api_key`, `reranker_top_k` |
| `agent/nodes/reranker.py` | No change — already calls factory |

---

## Phase 2: JWT Auth

**User model:** New `User` SQLAlchemy model in PostgreSQL.

| Field | Type | Notes |
|-------|------|-------|
| id | UUID | primary key |
| email | str | unique, indexed |
| username | str | unique |
| hashed_password | str | bcrypt via `passlib` |
| created_at | datetime | auto |
| is_active | bool | default true |

**Auth flow:**

```
POST /api/v1/auth/register  → create user, return tokens
POST /api/v1/auth/login     → verify password, return tokens
POST /api/v1/auth/refresh   → exchange refresh token for new access token

Access token:  JWT, 30min expiry, contains user_id + email
Refresh token: JWT, 7 day expiry, contains user_id only
```

**Document isolation:** Documents and chunks get a `user_id` foreign key. Users only see and query their own documents. Qdrant metadata filter by `user_id` at query time.

**Protected endpoints:** All existing endpoints except `/health` and `/auth/*` require `Authorization: Bearer <token>` header. A `get_current_user` FastAPI dependency handles validation.

**Frontend:** New login/register page. Token stored in localStorage. API client attaches token to all requests. Redirect to login on 401.

**Dependencies:** `pyjwt`, `passlib[bcrypt]`

**Files:**

| File | Change |
|------|--------|
| `models/user.py` | New — User model |
| `schemas/auth.py` | New — LoginRequest, TokenResponse, RegisterRequest |
| `services/auth.py` | New — register, login, verify, refresh |
| `api/auth.py` | New — auth router |
| `api/dependencies.py` | New — `get_current_user` dependency |
| `api/documents.py` | Add user dependency, filter by user_id |
| `api/chat.py` | Add user dependency, scope to user's docs |
| `models/document.py` | Add `user_id` FK |
| `core/config.py` | Add `jwt_secret`, `jwt_algorithm`, `access_token_expire_minutes` |
| `frontend/src/pages/Login.tsx` | New page |
| `frontend/src/api/client.ts` | Attach token header |
| `frontend/src/stores/authStore.ts` | New — token + user state |

---

## Phase 3: Prometheus + Grafana

**Metrics exposed from backend via `prometheus_client`:**

| Metric | Type | Labels | Where instrumented |
|--------|------|--------|--------------------|
| `llm_request_duration_seconds` | Histogram | provider, model | `BaseLLMClient` subclasses |
| `llm_tokens_total` | Counter | provider, model, type (prompt/completion) | `BaseLLMClient` subclasses |
| `llm_cost_dollars_total` | Counter | provider, model | `BaseLLMClient` subclasses |
| `http_request_duration_seconds` | Histogram | method, endpoint, status | FastAPI middleware |
| `ingestion_stage_duration_seconds` | Histogram | stage | `IngestionService` |
| `ingestion_documents_total` | Counter | status (success/fail) | `IngestionService` |
| `chunks_created_total` | Counter | type (parent/child) | `IngestionService` |
| `retrieval_duration_seconds` | Histogram | — | retriever node |
| `reranker_duration_seconds` | Histogram | strategy | reranker node |
| `celery_tasks_total` | Counter | task, status | Celery worker |
| `celery_queue_depth` | Gauge | — | Celery worker |

**Infrastructure (added to docker-compose):**

| Service | Port | Role |
|---------|------|------|
| `prometheus` | 9090 | Scrapes backend `/metrics` every 15s |
| `grafana` | 3002 | Pre-provisioned dashboards via JSON |

**Pre-built Grafana dashboards (provisioned via config):**
1. **API Health** — request rate, error rate, latency percentiles
2. **LLM Usage** — tokens/cost by provider, latency by model
3. **Pipeline** — ingestion throughput, stage durations, queue depth

**Dependency:** `prometheus_client`

**Files:**

| File | Change |
|------|--------|
| `core/metrics.py` | New — define all Prometheus metrics |
| `core/middleware.py` | New — HTTP request instrumentation middleware |
| `pipeline/base/llm_client.py` | Add metric recording in base class |
| `services/ingestion.py` | Add stage duration observations |
| `agent/nodes/retriever.py` | Add retrieval duration |
| `agent/nodes/reranker.py` | Add reranker duration |
| `main.py` | Mount `/metrics` endpoint, add middleware |
| `docker-compose.yml` | Add prometheus + grafana services |
| `infra/prometheus.yml` | New — scrape config |
| `infra/grafana/` | New — dashboard JSON + datasource provisioning |

---

## Phase 4: ChunkViewer Page

**Purpose:** New `/chunks` page in the React app to inspect how documents were parsed and chunked.

**Features:**
1. **Document selector** — dropdown of ingested documents
2. **Parent-child tree view** — expandable parents showing their children
3. **Chunk detail panel** — click to see:
   - Content (raw + enriched)
   - Metadata (type, page, section, word count, language)
   - Quality score
   - Tables: HTML rendering alongside markdown
   - Figures: image + generated description
4. **Filters** — by element type (text/table/figure/code), by page range
5. **Search** — text search across chunk content within a document

**API endpoints needed:**

| Endpoint | Method | Returns |
|----------|--------|---------|
| `GET /api/v1/documents/{id}/chunks` | GET | Parent chunks with nested children, metadata |
| `GET /api/v1/chunks/{chunk_id}` | GET | Single chunk full detail |

**Files:**

| File | Change |
|------|--------|
| `frontend/src/pages/Chunks.tsx` | New — main page layout |
| `frontend/src/components/chunks/ChunkTree.tsx` | New — parent-child tree |
| `frontend/src/components/chunks/ChunkDetail.tsx` | New — detail panel |
| `frontend/src/components/chunks/ChunkFilters.tsx` | New — type/page filters |
| `frontend/src/stores/chunkStore.ts` | New — Zustand store |
| `frontend/src/api/chunks.ts` | New — API client |
| `frontend/src/hooks/useChunks.ts` | New — data fetching hook |
| `backend/app/api/chunks.py` | New — chunk API router |
| `backend/app/main.py` | Register chunks router |
| `frontend/src/App.tsx` | Add `/chunks` route |
