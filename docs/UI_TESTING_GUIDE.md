# DocMind RAG — UI Testing Guide

> For browser automation agents testing at `http://localhost:3000`.
> Backend API base: `/api/v1` (proxied via Vite to `http://localhost:8000`).

---

## Prerequisites

```bash
# Start infrastructure
docker compose up -d qdrant postgres redis

# Start backend
uvicorn backend.app.main:app --reload

# Start Celery worker (required for document ingestion)
uv run celery -A backend.app.workers.celery_app worker --loglevel=info

# Start frontend
cd frontend && npm run dev   # runs on port 3000
```

### Test fixture

- PDF: `tests/fixtures/sample.pdf` ("Attention Is All You Need", 15 pages)
- Ingestion takes ~70 seconds (parsing + LLM enrichment + embedding)

---

## Page Map

```
Unauthenticated → Login page (only page visible)
Authenticated   → Nav: [Chat] [Documents] [Chunks]   [username] [dark/light] [logout]
```

| Page | URL state | Purpose |
|------|-----------|---------|
| Login | unauthenticated | Register / Sign in |
| Chat | `page=chat` (default) | Ask questions about documents |
| Documents | `page=documents` | Upload, list, delete documents |
| Chunks | `page=chunks` | Inspect parsed parent-child chunks |

---

## Flow 1: Authentication

### Register a new user

1. Navigate to `http://localhost:3000`
2. Page shows **Login** form (title: "Sign in to your account")
3. Click **"Register"** link at bottom → form switches to register mode
4. Fill in:
   - Email: `test@example.com`
   - Username: `testuser` (min 3 chars)
   - Password: `TestPass1234` (min 8 chars)
5. Click **"Create account"**
6. On success: redirected to **Chat** page, username shown in top-right
7. On failure: error banner appears (e.g. "Email or username already taken")

**API call:** `POST /api/v1/auth/register` → `{ access_token, refresh_token, user: { email, username } }`

### Login with existing user

1. On Login page, ensure mode is "Sign in" (default)
2. Fill in Email and Password
3. Click **"Sign in"**
4. On success: redirected to **Chat** page
5. On failure: "Invalid credentials" error banner

**API call:** `POST /api/v1/auth/login`

### Logout

1. Click the **logout icon** (arrow right icon, top-right corner)
2. Tokens cleared from localStorage
3. Redirected to Login page

### Session persistence

- Tokens stored in `localStorage` keys: `access_token`, `refresh_token`, `user`
- Page reload restores session via `loadFromStorage()`
- 401 responses auto-trigger logout

### Verify via curl

```bash
# Register
curl -s -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","username":"testuser","password":"TestPass1234"}'

# Login
curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"TestPass1234"}'
```

---

## Flow 2: Document Upload & Management

### Upload a document

1. Navigate to **Documents** page (click "Documents" in nav)
2. Page shows "0 documents in your knowledge base" for new users
3. Click **"+ Choose file"** button or drag-and-drop a file onto the upload zone
4. Accepted formats: `.pdf`, `.docx`, `.txt`, `.md` (max 50MB)
5. After upload:
   - Document appears in list with **"processing"** status (spinner)
   - Frontend polls `GET /api/v1/documents/task/{taskId}` every 2 seconds
   - When complete: status changes to **"Ready"** (green dot)
6. Click **"Refresh"** button to manually re-fetch document list

**API calls:**
- `POST /api/v1/documents/upload` (multipart form-data with `file` field)
- `GET /api/v1/documents/task/{taskId}` (polling)
- `GET /api/v1/documents/` (list)

### Upload via curl (for browser automation agents)

Browser automation tools cannot interact with native file picker dialogs. Upload via API instead:

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"TestPass1234"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -s -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@tests/fixtures/sample.pdf"
```

Then refresh the Documents page in the browser to see the uploaded document.

### Delete a document

1. Hover over a document card → **trash icon** appears
2. Click trash icon to delete
3. Document removed from list

**API call:** `DELETE /api/v1/documents/{docId}`

### Document card anatomy

```
┌─────────────────────────┐
│  [gradient header]      │
│     📄 (file icon)      │
├─────────────────────────┤
│  sample.pdf             │
│  ● Ready          PDF   │
│              [🗑 hover]  │
└─────────────────────────┘
```

- Status: green dot = Ready, amber pulsing = Processing
- Gradient colors: PDF=rose/orange, DOCX=blue/cyan, TXT=emerald/teal, MD=violet/purple

---

## Flow 3: Chunk Viewer

### View chunks for a document

1. Navigate to **Chunks** page
2. Select a document from the **dropdown** (only "ready" documents shown)
3. Left panel shows **Parent chunks** tree (e.g., "Parent chunks (33)")
4. Each row shows: `[expand] [TYPE_BADGE] Section Name    p.N  NNNw`
   - TYPE_BADGE colors: text=slate, table=blue, figure=purple, code=green
   - `p.N` = page number, `NNNw` = word count

### Select a chunk (view detail)

1. Click on any parent chunk row
2. **Left panel**: row highlights, children expand below (indented, with content preview)
3. **Right panel**: shows chunk detail:
   - Title (section name)
   - Children count badge (if applicable)
   - Metadata row: Type, Page, Section, Language, Word count
   - Content tabs: Raw (always), Markdown (if available), HTML (if available)

### Use filters

1. **Type filter**: dropdown "All" / "Text" / "Table" / "Figure" / "Code"
2. **Page filter**: numeric input
3. **Search**: text input — filters chunks by content

**API call:** `GET /api/v1/documents/{docId}/chunks?type_filter=&page_filter=&search=`

### Known behavior

- Parent chunks come from PostgreSQL, child chunks come from Qdrant
- Children display `content_raw` field (not `content`)
- Atomic elements (figures, tables) may appear as top-level items without children

---

## Flow 4: Chat (RAG Pipeline)

### Ask a question

1. Navigate to **Chat** page
2. Type a question in the textarea at the bottom
3. Press Enter or click **Send** button
4. User message appears right-aligned (purple bubble)
5. Assistant responds with streaming text (left-aligned, with avatar)
6. During streaming: **Stop** button appears, "Thinking..." indicator shows before first token

### Pipeline trace

- After response, expandable **"Pipeline trace (N steps)"** shows
- Click to expand: shows pipeline steps as connected badges
- Example steps: "Reranked: 5 → 3 chunks (strategy=cohere)"

### Sources

- If retrieval finds relevant chunks, **source badges** appear below the response
- Inline `[1]`, `[2]` citations in text are clickable → hover card shows source detail
- Source hover card: document name, page, section, content preview, confidence score bar

### Model selector

- Click **"GPT-4o"** dropdown (bottom-right of input area)
- Choose between "GPT-4o" (green) and "Claude" (amber)

### Clear chat

- Click **"Clear chat"** link at bottom to reset conversation

**API call:** `POST /api/v1/chat/` (SSE stream)

### Important: retrieval threshold

The retrieval score threshold is configured at `settings.retrieval_score_threshold` (default: 0.4). If the chat returns "I cannot determine..." with 0 chunks, the threshold may be too high for the embedding model's cosine similarity distribution. Lower to 0.25 in `.env` or `backend/app/core/config.py`.

---

## Flow 5: Dark Mode

1. Click the **sun/moon icon** in the top-right nav bar
2. Entire UI switches between light and dark themes
3. Toggle label changes: "Switch to dark mode" ↔ "Switch to light mode"
4. Dark mode applies `dark` class to `<html>` element
5. Note: dark mode preference is **not persisted** across page reloads (client-side only)

---

## Flow 6: Prometheus Metrics

### Verify metrics endpoint

```bash
curl -s http://localhost:8000/metrics/ | head -5
```

### Key custom metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `http_request_duration_seconds` | Histogram | endpoint, method, status | HTTP request latency |
| `llm_request_duration_seconds` | Histogram | model, operation | LLM API call latency |
| `llm_tokens_total` | Counter | model, type | Token usage (prompt/completion) |
| `ingestion_stage_duration_seconds` | Histogram | stage | Pipeline stage timing |
| `ingestion_documents_total` | Counter | status | Documents ingested |
| `chunks_created_total` | Counter | type | Chunks created by type |
| `retrieval_duration_seconds` | Histogram | — | Vector search latency |
| `reranker_duration_seconds` | Histogram | strategy | Reranking latency |

### Grafana (optional)

```bash
docker compose up -d prometheus grafana
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3002 (admin/admin)
```

---

## Interactive Elements Reference

### Login Page

| Element | Type | Selector hint |
|---------|------|---------------|
| Email input | `input[type=email]` | placeholder "you@example.com" |
| Username input | `input[type=text]` | placeholder "johndoe" (register mode only) |
| Password input | `input[type=password]` | placeholder "********" |
| Submit button | `button[type=submit]` | text "Sign in" or "Create account" |
| Mode toggle | `button[type=button]` | text "Register" or "Sign in" |

### Main App (authenticated)

| Element | Type | Selector hint |
|---------|------|---------------|
| Chat nav | `button` | text "Chat" |
| Documents nav | `button` | text "Documents" |
| Chunks nav | `button` | text "Chunks" |
| Dark mode toggle | `button` | text "Switch to dark/light mode" |
| Logout | `button` | text "Log out" |

### Chat Page

| Element | Type | Selector hint |
|---------|------|---------------|
| Message input | `textarea` | placeholder "Ask a question about your documents..." |
| Send button | `button[type=submit]` | text "Send message" |
| LLM selector | `button` | text "GPT-4o" or "Claude" |
| Clear chat | button | text "Clear chat" |
| Suggestion 1 | `button` | text "Summarize the key findings" |
| Suggestion 2 | `button` | text "What does the paper conclude?" |

### Documents Page

| Element | Type | Selector hint |
|---------|------|---------------|
| Choose file | `button` | text "+ Choose file" |
| Refresh | `button` | text "Refresh" |
| Delete (on card hover) | `button` | trash icon on document card |

### Chunks Page

| Element | Type | Selector hint |
|---------|------|---------------|
| Document selector | `select` / `combobox` | text "Select a document..." |
| Type filter | dropdown | text "All" |
| Page filter | input | text "Page" |
| Search | `input` | placeholder "Search chunk content..." |
| Parent chunk rows | `button` | contains type badge + section name |
| Child chunk rows | `button` | indented, contains content preview |

---

## Common Issues & Debugging

| Issue | Cause | Fix |
|-------|-------|-----|
| "Invalid credentials" | Wrong password or user doesn't exist | Register a new user or verify password via curl |
| Document stuck on "processing" | Celery worker not running or task not registered | Restart worker; check `celery_app.py` has `include` config |
| Documents page 500 error | Missing `user_id` column in DB | Run: `ALTER TABLE documents ADD COLUMN IF NOT EXISTS user_id VARCHAR` |
| Chunk Viewer crashes on expand | `child.content` undefined | Fixed: uses `content_raw \|\| content \|\| ""` |
| Chat returns "cannot determine" | Retrieval score threshold too high (0.4) | Lower `retrieval_score_threshold` to 0.25 |
| "Not Found" on auth endpoints | Wrong API prefix | Auth is at `/api/v1/auth/`, not `/api/auth/` |
| 401 auto-logout loop | Expired token, axios interceptor triggers logout | Re-login; tokens expire after 30 min |

---

## Quick Smoke Test Checklist

```
[ ] Register new user → lands on Chat page
[ ] Logout → returns to Login
[ ] Login with registered user → lands on Chat page
[ ] Navigate to Documents → shows 0 documents
[ ] Upload sample.pdf (via curl or UI) → processing → ready
[ ] Navigate to Chunks → select document → tree loads (33 parents)
[ ] Click a parent chunk → detail panel shows content + metadata
[ ] Expand parent → children visible with content previews
[ ] Navigate to Chat → type question → get streamed response
[ ] Toggle dark mode → UI switches theme
[ ] Check /metrics/ endpoint → custom histograms present
```
