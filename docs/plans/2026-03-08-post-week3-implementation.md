# Post-Week 3 Roadmap — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Cohere reranking, JWT authentication with user isolation, Prometheus + Grafana observability, and a ChunkViewer frontend page.

**Architecture:** Each phase builds on the previous. Cohere plugs into the existing `BaseReranker` ABC. JWT adds a `User` model and FastAPI dependency that protects all endpoints. Prometheus instruments LLM calls, pipeline stages, and HTTP middleware. ChunkViewer adds a new React page with backend API for chunk inspection.

**Tech Stack:** `cohere`, `pyjwt`, `passlib[bcrypt]`, `prometheus_client`, React + Zustand + Tailwind

---

## Phase 1: CohereReranker

### Task 1.1: Add Cohere Config

**Files:**
- Modify: `backend/app/core/config.py`
- Modify: `.env.example`

**Step 1: Add config fields**

Add to `Settings` in `backend/app/core/config.py`, after the retrieval section (line 31):

```python
# Reranker
reranker_strategy: str = "identity"
cohere_api_key: str = ""
reranker_top_n: int = 5
```

**Step 2: Update .env.example**

Add after existing keys:

```
COHERE_API_KEY=
RERANKER_STRATEGY=identity
```

**Step 3: Commit**

```bash
git add backend/app/core/config.py .env.example
git commit -m "feat: add reranker config (strategy, cohere key, top_n)"
```

---

### Task 1.2: Implement CohereReranker

**Files:**
- Create: `backend/app/pipeline/rerankers/cohere_reranker.py`
- Test: `tests/unit/pipeline/test_cohere_reranker.py`

**Step 1: Write the failing test**

```python
# tests/unit/pipeline/test_cohere_reranker.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_cohere():
    with patch("backend.app.pipeline.rerankers.cohere_reranker.cohere") as mock:
        client = MagicMock()
        mock.ClientV2.return_value = client
        yield client


def _make_chunks(n: int) -> list[dict]:
    return [
        {"content": f"chunk {i}", "score": 1 - i * 0.05, "doc_name": "test.pdf"}
        for i in range(n)
    ]


async def test_cohere_reranker_returns_top_n(mock_cohere):
    from backend.app.pipeline.rerankers.cohere_reranker import CohereReranker

    # Mock Cohere response
    mock_results = [MagicMock(index=i, relevance_score=1 - i * 0.1) for i in range(3)]
    mock_cohere.rerank.return_value = MagicMock(results=mock_results)

    reranker = CohereReranker(api_key="test-key")
    chunks = _make_chunks(10)
    result = await reranker.rerank("revenue question", chunks, top_n=3)

    assert len(result) == 3
    mock_cohere.rerank.assert_called_once()


async def test_cohere_reranker_preserves_chunk_data(mock_cohere):
    from backend.app.pipeline.rerankers.cohere_reranker import CohereReranker

    mock_results = [MagicMock(index=2, relevance_score=0.95)]
    mock_cohere.rerank.return_value = MagicMock(results=mock_results)

    reranker = CohereReranker(api_key="test-key")
    chunks = _make_chunks(5)
    result = await reranker.rerank("query", chunks, top_n=1)

    assert result[0]["content"] == "chunk 2"
    assert result[0]["score"] == 0.95  # updated to Cohere score


async def test_cohere_reranker_falls_back_on_error(mock_cohere):
    from backend.app.pipeline.rerankers.cohere_reranker import CohereReranker

    mock_cohere.rerank.side_effect = Exception("API error")

    reranker = CohereReranker(api_key="test-key")
    chunks = _make_chunks(10)
    result = await reranker.rerank("query", chunks, top_n=5)

    # Falls back to identity — returns first top_n unchanged
    assert len(result) == 5
    assert result[0]["content"] == "chunk 0"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/pipeline/test_cohere_reranker.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Install cohere and write implementation**

```bash
uv add cohere
```

```python
# backend/app/pipeline/rerankers/cohere_reranker.py
import cohere

from backend.app.core.logging import logger
from backend.app.pipeline.base.reranker import BaseReranker


class CohereReranker(BaseReranker):
    """Cross-encoder reranker using Cohere Rerank v3."""

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v3.0",
    ):
        self._client = cohere.ClientV2(api_key=api_key)
        self._model = model

    async def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_n: int,
    ) -> list[dict]:
        if not chunks:
            return []

        documents = [c.get("content", "") for c in chunks]

        try:
            response = self._client.rerank(
                model=self._model,
                query=query,
                documents=documents,
                top_n=top_n,
            )
        except Exception as e:
            logger.warning("cohere_rerank_failed", error=str(e))
            return chunks[:top_n]

        reranked = []
        for result in response.results:
            chunk = {**chunks[result.index]}
            chunk["score"] = result.relevance_score
            reranked.append(chunk)

        return reranked
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/pipeline/test_cohere_reranker.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add backend/app/pipeline/rerankers/cohere_reranker.py tests/unit/pipeline/test_cohere_reranker.py pyproject.toml uv.lock
git commit -m "feat: CohereReranker with fallback to identity on error"
```

---

### Task 1.3: Wire Cohere Into Factory + Agent Node

**Files:**
- Modify: `backend/app/pipeline/rerankers/factory.py`
- Modify: `backend/app/agent/nodes/reranker.py`
- Test: `tests/unit/agent/test_reranker.py` (extend)

**Step 1: Write the failing test**

Add to `tests/unit/agent/test_reranker.py`:

```python
from unittest.mock import patch, AsyncMock


async def test_reranker_node_uses_configured_strategy():
    chunks = [{"content": f"chunk {i}", "score": 0.9} for i in range(10)]
    state = _make_state(chunks=chunks)

    mock_reranker = AsyncMock()
    mock_reranker.rerank.return_value = chunks[:3]

    with patch(
        "backend.app.agent.nodes.reranker.RerankerFactory"
    ) as mock_factory:
        mock_factory.create.return_value = mock_reranker
        result = await reranker_node(state)
        mock_factory.create.assert_called_once()  # Uses factory, not hardcoded
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/agent/test_reranker.py::test_reranker_node_uses_configured_strategy -v`
Expected: PASS (factory is already used — this confirms existing behavior)

**Step 3: Update factory**

```python
# backend/app/pipeline/rerankers/factory.py
from backend.app.core.config import settings
from backend.app.pipeline.base.reranker import BaseReranker
from backend.app.pipeline.rerankers.identity_reranker import IdentityReranker


class RerankerFactory:

    @staticmethod
    def create(strategy: str = None) -> BaseReranker:
        strategy = strategy or settings.reranker_strategy

        if strategy == "cohere":
            from backend.app.pipeline.rerankers.cohere_reranker import CohereReranker
            return CohereReranker(api_key=settings.cohere_api_key)

        if strategy == "identity":
            return IdentityReranker()

        raise ValueError(
            f"Unknown reranker: {strategy}. Choose: identity, cohere"
        )
```

**Step 4: Update agent node to use config strategy and dynamic trace**

```python
# backend/app/agent/nodes/reranker.py
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

    strategy = settings.reranker_strategy
    reranker = RerankerFactory.create(strategy)
    chunks = state.get("retrieved_chunks", [])

    reranked = await reranker.rerank(
        query=state["original_query"],
        chunks=chunks,
        top_n=settings.reranker_top_n,
    )

    log.info("reranked", before=len(chunks), after=len(reranked), strategy=strategy)

    return {
        "reranked_chunks": reranked,
        "agent_trace": [
            f"Reranked: {len(chunks)} → {len(reranked)} chunks "
            f"(strategy={strategy})"
        ],
    }
```

**Step 5: Run all reranker tests**

Run: `pytest tests/unit/agent/test_reranker.py tests/unit/pipeline/test_cohere_reranker.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add backend/app/pipeline/rerankers/factory.py backend/app/agent/nodes/reranker.py tests/unit/agent/test_reranker.py
git commit -m "feat: wire CohereReranker into factory + agent node via config"
```

---

## Phase 2: JWT Auth

### Task 2.1: Add Auth Dependencies + Config

**Files:**
- Modify: `pyproject.toml`
- Modify: `backend/app/core/config.py`
- Modify: `.env.example`

**Step 1: Install dependencies**

```bash
uv add pyjwt passlib[bcrypt]
```

**Step 2: Add config fields**

Add to `Settings` in `backend/app/core/config.py`:

```python
# Auth
jwt_secret: str = "change-me-in-production"
jwt_algorithm: str = "HS256"
access_token_expire_minutes: int = 30
refresh_token_expire_days: int = 7
```

**Step 3: Update .env.example**

```
JWT_SECRET=change-me-in-production
```

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock backend/app/core/config.py .env.example
git commit -m "feat: add JWT auth dependencies and config"
```

---

### Task 2.2: User Model

**Files:**
- Create: `backend/app/models/user.py`
- Modify: `backend/app/core/database.py`

**Step 1: Create User model**

```python
# backend/app/models/user.py
import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, String

from backend.app.models.document import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
```

**Step 2: Import in database.py so table is created**

Add import in `backend/app/core/database.py` after `from backend.app.models.document import Base`:

```python
import backend.app.models.user  # noqa: F401 — register User table
```

**Step 3: Commit**

```bash
git add backend/app/models/user.py backend/app/core/database.py
git commit -m "feat: User model (email, username, hashed_password)"
```

---

### Task 2.3: Auth Schemas

**Files:**
- Create: `backend/app/schemas/auth.py`

**Step 1: Create schemas**

```python
# backend/app/schemas/auth.py
from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: str = Field(min_length=5)
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    is_active: bool
```

**Step 2: Commit**

```bash
git add backend/app/schemas/auth.py
git commit -m "feat: auth Pydantic schemas (register, login, token, user)"
```

---

### Task 2.4: Auth Service

**Files:**
- Create: `backend/app/services/auth.py`
- Test: `tests/unit/test_auth_service.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_auth_service.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.services.auth import AuthService


@pytest.fixture
def auth_service():
    return AuthService()


def test_hash_password(auth_service):
    hashed = auth_service.hash_password("testpassword")
    assert hashed != "testpassword"
    assert auth_service.verify_password("testpassword", hashed)


def test_verify_wrong_password(auth_service):
    hashed = auth_service.hash_password("correct")
    assert not auth_service.verify_password("wrong", hashed)


def test_create_access_token(auth_service):
    token = auth_service.create_access_token(user_id="user-1", email="a@b.com")
    assert isinstance(token, str)
    assert len(token) > 0


def test_create_refresh_token(auth_service):
    token = auth_service.create_refresh_token(user_id="user-1")
    assert isinstance(token, str)


def test_decode_valid_token(auth_service):
    token = auth_service.create_access_token(user_id="user-1", email="a@b.com")
    payload = auth_service.decode_token(token)
    assert payload["user_id"] == "user-1"
    assert payload["email"] == "a@b.com"
    assert payload["type"] == "access"


def test_decode_expired_token(auth_service):
    import jwt
    from datetime import datetime, timedelta
    from backend.app.core.config import settings

    payload = {
        "user_id": "user-1",
        "type": "access",
        "exp": datetime.utcnow() - timedelta(hours=1),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    result = auth_service.decode_token(token)
    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_auth_service.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# backend/app/services/auth.py
from datetime import datetime, timedelta

import jwt
from passlib.context import CryptContext
from sqlalchemy import select

from backend.app.core.config import settings
from backend.app.core.database import AsyncSessionLocal
from backend.app.core.logging import logger
from backend.app.models.user import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:

    def hash_password(self, password: str) -> str:
        return pwd_context.hash(password)

    def verify_password(self, plain: str, hashed: str) -> bool:
        return pwd_context.verify(plain, hashed)

    def create_access_token(self, user_id: str, email: str) -> str:
        payload = {
            "user_id": user_id,
            "email": email,
            "type": "access",
            "exp": datetime.utcnow()
            + timedelta(minutes=settings.access_token_expire_minutes),
        }
        return jwt.encode(
            payload, settings.jwt_secret, algorithm=settings.jwt_algorithm
        )

    def create_refresh_token(self, user_id: str) -> str:
        payload = {
            "user_id": user_id,
            "type": "refresh",
            "exp": datetime.utcnow()
            + timedelta(days=settings.refresh_token_expire_days),
        }
        return jwt.encode(
            payload, settings.jwt_secret, algorithm=settings.jwt_algorithm
        )

    def decode_token(self, token: str) -> dict | None:
        try:
            return jwt.decode(
                token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
            )
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return None

    async def register(self, email: str, username: str, password: str) -> User:
        async with AsyncSessionLocal() as session:
            # Check existing
            existing = await session.execute(
                select(User).where((User.email == email) | (User.username == username))
            )
            if existing.scalar_one_or_none():
                raise ValueError("Email or username already taken")

            user = User(
                email=email,
                username=username,
                hashed_password=self.hash_password(password),
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            logger.info("user_registered", user_id=user.id, email=email)
            return user

    async def authenticate(self, email: str, password: str) -> User | None:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(User).where(User.email == email)
            )
            user = result.scalar_one_or_none()
            if not user or not self.verify_password(password, user.hashed_password):
                return None
            return user

    async def get_user_by_id(self, user_id: str) -> User | None:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_auth_service.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add backend/app/services/auth.py tests/unit/test_auth_service.py
git commit -m "feat: AuthService — hash, JWT, register, authenticate"
```

---

### Task 2.5: Auth Dependency + API Router

**Files:**
- Create: `backend/app/api/dependencies.py`
- Create: `backend/app/api/auth.py`
- Modify: `backend/app/main.py`
- Test: `tests/unit/test_auth_api.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_auth_api.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from backend.app.api.dependencies import get_current_user


async def test_get_current_user_no_token():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await get_current_user(authorization=None)
    assert exc.value.status_code == 401


async def test_get_current_user_invalid_token():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await get_current_user(authorization="Bearer invalid-token")
    assert exc.value.status_code == 401
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_auth_api.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write dependencies**

```python
# backend/app/api/dependencies.py
from fastapi import Header, HTTPException

from backend.app.services.auth import AuthService

_auth = AuthService()


async def get_current_user(
    authorization: str | None = Header(default=None),
) -> dict:
    """Extract and validate JWT from Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization.removeprefix("Bearer ").strip()
    payload = _auth.decode_token(token)

    if not payload or payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = await _auth.get_user_by_id(payload["user_id"])
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return {"user_id": user.id, "email": user.email, "username": user.username}
```

**Step 4: Write auth router**

```python
# backend/app/api/auth.py
from fastapi import APIRouter, HTTPException

from backend.app.schemas.auth import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from backend.app.services.auth import AuthService

router = APIRouter()
_auth = AuthService()


@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest):
    try:
        user = await _auth.register(
            email=request.email,
            username=request.username,
            password=request.password,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return TokenResponse(
        access_token=_auth.create_access_token(user.id, user.email),
        refresh_token=_auth.create_refresh_token(user.id),
    )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    user = await _auth.authenticate(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return TokenResponse(
        access_token=_auth.create_access_token(user.id, user.email),
        refresh_token=_auth.create_refresh_token(user.id),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(request: RefreshRequest):
    payload = _auth.decode_token(request.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user = await _auth.get_user_by_id(payload["user_id"])
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found")

    return TokenResponse(
        access_token=_auth.create_access_token(user.id, user.email),
        refresh_token=_auth.create_refresh_token(user.id),
    )
```

**Step 5: Register auth router in main.py**

Add to `backend/app/main.py` imports:

```python
from backend.app.api import auth, chat, documents, eval, health
```

Add after the health router:

```python
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
```

**Step 6: Run tests**

Run: `pytest tests/unit/test_auth_api.py tests/unit/test_auth_service.py -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add backend/app/api/dependencies.py backend/app/api/auth.py backend/app/main.py tests/unit/test_auth_api.py
git commit -m "feat: auth API — register, login, refresh + get_current_user dependency"
```

---

### Task 2.6: Protect Existing Endpoints + Document Isolation

**Files:**
- Modify: `backend/app/models/document.py` — add `user_id` column
- Modify: `backend/app/api/documents.py` — add auth dependency, filter by user
- Modify: `backend/app/api/chat.py` — add auth dependency, scope to user's docs
- Modify: `backend/app/services/ingestion.py` — accept `user_id` parameter
- Modify: `backend/app/vectorstore/qdrant_client.py` — add `user_id` to filter builder

**Step 1: Add user_id to Document model**

In `backend/app/models/document.py`, add to `Document` class after `doc_name`:

```python
user_id = Column(String, nullable=True, index=True)  # nullable for migration
```

Add to `ParentChunk` after `doc_id`:

```python
user_id = Column(String, nullable=True, index=True)
```

**Step 2: Update documents API**

```python
# backend/app/api/documents.py
import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.app.api.dependencies import get_current_user
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
    user: dict = Depends(get_current_user),
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
        user_id=user["user_id"],
    )

    task = ingest_document_task.delay(
        file_path=tmp_path,
        doc_name=file.filename,
        language=language,
        parser_strategy=parser_strategy,
        user_id=user["user_id"],
    )

    return JSONResponse(
        {
            "status": "processing",
            "task_id": task.id,
            "message": f"Document '{file.filename}' queued for processing",
        }
    )


@router.get("/task/{task_id}")
async def get_task_status(
    task_id: str,
    user: dict = Depends(get_current_user),
):
    task = ingest_document_task.AsyncResult(task_id)
    if task.state == "SUCCESS":
        return {"status": "ready", "result": task.result}
    elif task.state == "FAILURE":
        return {"status": "failed", "error": str(task.result)}
    else:
        return {"status": task.state.lower()}


@router.get("/")
async def list_documents(user: dict = Depends(get_current_user)):
    from sqlalchemy import select

    from backend.app.core.database import AsyncSessionLocal
    from backend.app.models.document import Document

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Document).where(Document.user_id == user["user_id"])
        )
        docs = result.scalars().all()
    return {
        "documents": [
            {"doc_id": d.doc_id, "doc_name": d.doc_name, "status": d.status}
            for d in docs
        ],
        "total": len(docs),
    }


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    user: dict = Depends(get_current_user),
):
    from sqlalchemy import select

    from backend.app.core.database import AsyncSessionLocal
    from backend.app.models.document import Document

    # Verify ownership
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Document).where(
                Document.doc_id == doc_id,
                Document.user_id == user["user_id"],
            )
        )
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Document not found")

    from backend.app.services.ingestion import IngestionService

    service = IngestionService()
    await service.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}
```

**Step 3: Update chat API**

In `backend/app/api/chat.py`, add auth dependency:

```python
from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from backend.app.api.dependencies import get_current_user
from backend.app.core.logging import logger
from backend.app.schemas.chat import ChatRequest
from backend.app.services.rag import RAGService

router = APIRouter()

_rag_service = RAGService()


@router.post("/")
async def chat(
    request: ChatRequest,
    user: dict = Depends(get_current_user),
):
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
    async for event in _rag_service.stream_query(request):
        yield {"data": event}
```

**Step 4: Update IngestionService to accept user_id**

In `backend/app/services/ingestion.py`, update `ingest()` signature:

```python
async def ingest(
    self,
    file_path: str,
    doc_name: str,
    language: str = "en",
    parser_strategy: str = "auto",
    user_id: str = None,
) -> dict:
```

In `_store_parents()`, pass `user_id` to Document and ParentChunk:

```python
doc = Document(
    doc_id=doc_id,
    doc_name=doc_name,
    user_id=user_id,
    ...
)
```

```python
pg_chunk = ParentChunk(
    chunk_id=chunk.chunk_id,
    doc_id=doc_id,
    user_id=user_id,
    ...
)
```

Also update `_store_parents` signature to accept `user_id`:

```python
async def _store_parents(self, doc_id, doc_name, parent_chunks, doc_metadata, user_id=None):
```

And update the call site in `ingest()`:

```python
await self._store_parents(doc_id, doc_name, parent_chunks, doc_metadata, user_id=user_id)
```

**Step 5: Update Celery task to pass user_id**

In `backend/app/workers/ingest_tasks.py`, add `user_id` parameter and pass it through to `IngestionService.ingest()`.

**Step 6: Add user_id to Qdrant filter builder**

In `backend/app/vectorstore/qdrant_client.py`, add to `_build_filter()`:

```python
if "user_id" in filters and filters["user_id"]:
    conditions.append(
        FieldCondition(
            key="user_id", match=MatchValue(value=filters["user_id"])
        )
    )
```

**Step 7: Run full test suite**

Run: `pytest tests/unit/ -v`
Expected: ALL PASS (existing tests unaffected — `get_current_user` only triggers on real HTTP requests)

**Step 8: Commit**

```bash
git add backend/app/models/document.py backend/app/api/documents.py backend/app/api/chat.py backend/app/services/ingestion.py backend/app/vectorstore/qdrant_client.py backend/app/workers/ingest_tasks.py
git commit -m "feat: protect endpoints with JWT, add user_id isolation to documents"
```

---

### Task 2.7: Frontend Auth (Login Page + Token Management)

**Files:**
- Create: `frontend/src/stores/authStore.ts`
- Create: `frontend/src/pages/Login.tsx`
- Modify: `frontend/src/api/client.ts` — attach token
- Modify: `frontend/src/App.tsx` — add login page + auth guard

**Step 1: Create auth store**

```typescript
// frontend/src/stores/authStore.ts
import { create } from "zustand";

interface AuthState {
  token: string | null;
  refreshToken: string | null;
  user: { email: string; username: string } | null;
  isAuthenticated: boolean;

  login: (token: string, refreshToken: string, user: { email: string; username: string }) => void;
  logout: () => void;
  loadFromStorage: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  token: null,
  refreshToken: null,
  user: null,
  isAuthenticated: false,

  login: (token, refreshToken, user) => {
    localStorage.setItem("access_token", token);
    localStorage.setItem("refresh_token", refreshToken);
    localStorage.setItem("user", JSON.stringify(user));
    set({ token, refreshToken, user, isAuthenticated: true });
  },

  logout: () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    localStorage.removeItem("user");
    set({ token: null, refreshToken: null, user: null, isAuthenticated: false });
  },

  loadFromStorage: () => {
    const token = localStorage.getItem("access_token");
    const refreshToken = localStorage.getItem("refresh_token");
    const userStr = localStorage.getItem("user");
    if (token && userStr) {
      set({
        token,
        refreshToken,
        user: JSON.parse(userStr),
        isAuthenticated: true,
      });
    }
  },
}));
```

**Step 2: Update API client to attach token**

```typescript
// frontend/src/api/client.ts
import axios from "axios";
import { useAuthStore } from "@/stores/authStore";

const api = axios.create({
  baseURL: "/api/v1",
  headers: { "Content-Type": "application/json" },
});

api.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout();
    }
    return Promise.reject(error);
  },
);

export default api;
```

**Step 3: Create Login page**

Create `frontend/src/pages/Login.tsx` — a form with email, username (for register), password. Toggle between login and register mode. On success, call `authStore.login()` and redirect to chat.

Use the existing Glass Observatory styling (frosted glass panels, violet-blue theme) to match the rest of the app. Include:
- Email + password fields for login
- Email + username + password for register
- Toggle link between modes
- Error display
- Call `POST /api/v1/auth/login` or `/register`

**Step 4: Update App.tsx — add auth guard**

Update `App.tsx`:
- Import `useAuthStore` and `Login`
- On mount, call `loadFromStorage()`
- If not authenticated, render `Login` instead of main content
- Add `"chunks"` to the `Page` type and nav pills (for Phase 4)
- Add logout button in nav bar

**Step 5: Update SSE chat client to include token**

In `frontend/src/api/chat.ts`, add the Authorization header to the `fetch()` call for SSE streaming (since axios is not used for SSE):

```typescript
const token = localStorage.getItem("access_token");
const headers: Record<string, string> = { "Content-Type": "application/json" };
if (token) headers["Authorization"] = `Bearer ${token}`;
```

**Step 6: Commit**

```bash
git add frontend/src/stores/authStore.ts frontend/src/pages/Login.tsx frontend/src/api/client.ts frontend/src/api/chat.ts frontend/src/App.tsx
git commit -m "feat: frontend auth — login/register page, token management, auth guard"
```

---

## Phase 3: Prometheus + Grafana

### Task 3.1: Define Prometheus Metrics

**Files:**
- Create: `backend/app/core/metrics.py`
- Test: `tests/unit/test_metrics.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_metrics.py
from backend.app.core.metrics import (
    LLM_REQUEST_DURATION,
    LLM_TOKENS_TOTAL,
    HTTP_REQUEST_DURATION,
    INGESTION_STAGE_DURATION,
    CHUNKS_CREATED_TOTAL,
)


def test_metrics_are_defined():
    assert LLM_REQUEST_DURATION is not None
    assert LLM_TOKENS_TOTAL is not None
    assert HTTP_REQUEST_DURATION is not None
    assert INGESTION_STAGE_DURATION is not None
    assert CHUNKS_CREATED_TOTAL is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Install and implement**

```bash
uv add prometheus_client
```

```python
# backend/app/core/metrics.py
from prometheus_client import Counter, Gauge, Histogram

# LLM metrics
LLM_REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency",
    labelnames=["provider", "model"],
    buckets=[0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total LLM tokens",
    labelnames=["provider", "model", "type"],
)

LLM_COST_TOTAL = Counter(
    "llm_cost_dollars_total",
    "Estimated LLM cost in USD",
    labelnames=["provider", "model"],
)

# HTTP metrics
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    labelnames=["method", "endpoint", "status"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Pipeline metrics
INGESTION_STAGE_DURATION = Histogram(
    "ingestion_stage_duration_seconds",
    "Duration of each ingestion stage",
    labelnames=["stage"],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
)

INGESTION_DOCUMENTS_TOTAL = Counter(
    "ingestion_documents_total",
    "Total documents ingested",
    labelnames=["status"],
)

CHUNKS_CREATED_TOTAL = Counter(
    "chunks_created_total",
    "Total chunks created",
    labelnames=["type"],
)

# Retrieval metrics
RETRIEVAL_DURATION = Histogram(
    "retrieval_duration_seconds",
    "Vector search latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

RERANKER_DURATION = Histogram(
    "reranker_duration_seconds",
    "Reranker latency",
    labelnames=["strategy"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

# Celery metrics
CELERY_TASKS_TOTAL = Counter(
    "celery_tasks_total",
    "Total Celery tasks",
    labelnames=["task", "status"],
)

CELERY_QUEUE_DEPTH = Gauge(
    "celery_queue_depth",
    "Current Celery queue depth",
)
```

**Step 4: Run test**

Run: `pytest tests/unit/test_metrics.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/core/metrics.py tests/unit/test_metrics.py pyproject.toml uv.lock
git commit -m "feat: define Prometheus metrics for LLM, HTTP, pipeline, retrieval"
```

---

### Task 3.2: HTTP Middleware + /metrics Endpoint

**Files:**
- Create: `backend/app/core/middleware.py`
- Modify: `backend/app/main.py`

**Step 1: Write middleware**

```python
# backend/app/core/middleware.py
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from backend.app.core.metrics import HTTP_REQUEST_DURATION


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        # Skip /metrics itself to avoid recursion
        if request.url.path != "/metrics":
            # Normalize path: replace UUIDs/IDs with {id}
            path = request.url.path
            HTTP_REQUEST_DURATION.labels(
                method=request.method,
                endpoint=path,
                status=response.status_code,
            ).observe(duration)

        return response
```

**Step 2: Update main.py**

Add to imports:

```python
from prometheus_client import make_asgi_app
from backend.app.core.middleware import PrometheusMiddleware
```

After the CORS middleware:

```python
app.add_middleware(PrometheusMiddleware)

# Mount /metrics for Prometheus scraping
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

**Step 3: Commit**

```bash
git add backend/app/core/middleware.py backend/app/main.py
git commit -m "feat: Prometheus HTTP middleware + /metrics endpoint"
```

---

### Task 3.3: Instrument LLM Clients

**Files:**
- Modify: `backend/app/pipeline/llm/openai_client.py`
- Modify: `backend/app/pipeline/llm/claude_client.py`

**Step 1: Add metrics to OpenAI client**

In the `complete()` method of `OpenAIClient`, wrap the API call with timing and token counting:

```python
import time
from backend.app.core.metrics import LLM_REQUEST_DURATION, LLM_TOKENS_TOTAL

# Before the call:
start = time.perf_counter()

# After the call:
duration = time.perf_counter() - start
LLM_REQUEST_DURATION.labels(provider="openai", model=self._model).observe(duration)
if hasattr(response, "usage") and response.usage:
    LLM_TOKENS_TOTAL.labels(provider="openai", model=self._model, type="prompt").inc(response.usage.prompt_tokens)
    LLM_TOKENS_TOTAL.labels(provider="openai", model=self._model, type="completion").inc(response.usage.completion_tokens)
```

Apply the same pattern to `stream()` and `complete_with_vision()`.

**Step 2: Add metrics to Claude client**

Same pattern for `ClaudeClient.complete()` and `stream()`, using `provider="anthropic"`.

**Step 3: Run existing LLM tests**

Run: `pytest tests/unit/pipeline/test_llm.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add backend/app/pipeline/llm/openai_client.py backend/app/pipeline/llm/claude_client.py
git commit -m "feat: instrument LLM clients with Prometheus metrics (latency, tokens)"
```

---

### Task 3.4: Instrument Pipeline + Agent Nodes

**Files:**
- Modify: `backend/app/services/ingestion.py` — add stage duration metrics
- Modify: `backend/app/agent/nodes/retriever.py` — add retrieval duration
- Modify: `backend/app/agent/nodes/reranker.py` — add reranker duration

**Step 1: Instrument ingestion stages**

In `backend/app/services/ingestion.py`, add timing around each stage:

```python
import time
from backend.app.core.metrics import (
    INGESTION_STAGE_DURATION,
    INGESTION_DOCUMENTS_TOTAL,
    CHUNKS_CREATED_TOTAL,
)

# Around each stage:
start = time.perf_counter()
# ... stage code ...
INGESTION_STAGE_DURATION.labels(stage="parse").observe(time.perf_counter() - start)

# At end of successful ingest:
INGESTION_DOCUMENTS_TOTAL.labels(status="success").inc()
CHUNKS_CREATED_TOTAL.labels(type="parent").inc(len(parent_chunks))
CHUNKS_CREATED_TOTAL.labels(type="child").inc(len(final_children))

# In except block:
INGESTION_DOCUMENTS_TOTAL.labels(status="fail").inc()
```

**Step 2: Instrument retriever node**

In `backend/app/agent/nodes/retriever.py`:

```python
import time
from backend.app.core.metrics import RETRIEVAL_DURATION

# Around the search loop:
start = time.perf_counter()
# ... existing retrieval code ...
RETRIEVAL_DURATION.observe(time.perf_counter() - start)
```

**Step 3: Instrument reranker node**

In `backend/app/agent/nodes/reranker.py`:

```python
import time
from backend.app.core.metrics import RERANKER_DURATION

start = time.perf_counter()
reranked = await reranker.rerank(...)
RERANKER_DURATION.labels(strategy=strategy).observe(time.perf_counter() - start)
```

**Step 4: Run tests**

Run: `pytest tests/unit/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add backend/app/services/ingestion.py backend/app/agent/nodes/retriever.py backend/app/agent/nodes/reranker.py
git commit -m "feat: instrument pipeline stages + agent nodes with Prometheus metrics"
```

---

### Task 3.5: Docker Compose — Prometheus + Grafana

**Files:**
- Create: `infra/prometheus.yml`
- Create: `infra/grafana/provisioning/datasources/prometheus.yml`
- Create: `infra/grafana/provisioning/dashboards/dashboard.yml`
- Create: `infra/grafana/dashboards/docmind-overview.json`
- Modify: `docker-compose.yml`

**Step 1: Create Prometheus config**

```yaml
# infra/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "docmind-backend"
    static_configs:
      - targets: ["backend:8000"]
    metrics_path: /metrics
```

**Step 2: Create Grafana datasource provisioning**

```yaml
# infra/grafana/provisioning/datasources/prometheus.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

**Step 3: Create Grafana dashboard provisioning**

```yaml
# infra/grafana/provisioning/dashboards/dashboard.yml
apiVersion: 1
providers:
  - name: "default"
    orgId: 1
    folder: ""
    type: file
    options:
      path: /var/lib/grafana/dashboards
```

**Step 4: Create overview dashboard JSON**

Create `infra/grafana/dashboards/docmind-overview.json` — a Grafana dashboard with panels for:
- HTTP request rate + error rate (from `http_request_duration_seconds`)
- LLM latency by provider (from `llm_request_duration_seconds`)
- Token usage by model (from `llm_tokens_total`)
- Ingestion throughput (from `ingestion_documents_total`)
- Pipeline stage durations (from `ingestion_stage_duration_seconds`)
- Retrieval latency (from `retrieval_duration_seconds`)

**Step 5: Add services to docker-compose.yml**

Add after `frontend` service:

```yaml
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./infra/prometheus.yml:/etc/prometheus/prometheus.yml
    depends_on:
      - backend

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3002:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=docmind
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
    volumes:
      - ./infra/grafana/provisioning:/etc/grafana/provisioning
      - ./infra/grafana/dashboards:/var/lib/grafana/dashboards
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
```

Add to volumes:

```yaml
  grafana_data:
```

**Step 6: Commit**

```bash
git add infra/ docker-compose.yml
git commit -m "feat: Prometheus + Grafana docker services with pre-provisioned dashboards"
```

---

## Phase 4: ChunkViewer Page

### Task 4.1: Chunks API Endpoint

**Files:**
- Create: `backend/app/api/chunks.py`
- Modify: `backend/app/main.py`
- Test: `tests/unit/test_chunks_api.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_chunks_api.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


async def test_get_document_chunks_returns_parents_with_children():
    """Chunks API should return parent chunks with nested children."""
    from backend.app.api.chunks import _build_chunk_tree

    parents = [
        MagicMock(
            chunk_id="p1",
            doc_id="doc1",
            content_raw="parent content",
            content_markdown="# parent",
            content_html="<p>parent</p>",
            type="text",
            page=1,
            section="Introduction",
            language="en",
            word_count=100,
            metadata_={},
        )
    ]

    children = [
        {
            "chunk_id": "c1",
            "parent_id": "p1",
            "content": "child enriched",
            "content_raw": "child raw",
            "type": "text",
            "page": 1,
            "section": "Introduction",
        },
        {
            "chunk_id": "c2",
            "parent_id": "p1",
            "content": "child 2 enriched",
            "content_raw": "child 2 raw",
            "type": "text",
            "page": 1,
            "section": "Introduction",
        },
    ]

    tree = _build_chunk_tree(parents, children)
    assert len(tree) == 1
    assert tree[0]["chunk_id"] == "p1"
    assert len(tree[0]["children"]) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_chunks_api.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# backend/app/api/chunks.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select

from backend.app.api.dependencies import get_current_user
from backend.app.core.database import AsyncSessionLocal
from backend.app.models.document import Document, ParentChunk
from backend.app.vectorstore.qdrant_client import QdrantWrapper

router = APIRouter()


def _build_chunk_tree(parents: list, children: list[dict]) -> list[dict]:
    """Build parent-child tree from flat lists."""
    tree = []
    children_by_parent = {}
    for child in children:
        pid = child.get("parent_id", "")
        children_by_parent.setdefault(pid, []).append(child)

    for parent in parents:
        tree.append({
            "chunk_id": parent.chunk_id,
            "content_raw": parent.content_raw,
            "content_markdown": parent.content_markdown,
            "content_html": parent.content_html,
            "type": parent.type,
            "page": parent.page,
            "section": parent.section,
            "language": parent.language,
            "word_count": parent.word_count,
            "children": children_by_parent.get(parent.chunk_id, []),
        })

    # Add orphan children (atomic chunks with no parent)
    orphan_parent_ids = set(children_by_parent.keys()) - {p.chunk_id for p in parents}
    for pid in orphan_parent_ids:
        if not pid:  # empty parent_id = atomic
            for child in children_by_parent[pid]:
                tree.append({
                    "chunk_id": child["chunk_id"],
                    "content_raw": child.get("content_raw", ""),
                    "content_markdown": child.get("content_markdown"),
                    "content_html": child.get("content_html"),
                    "type": child.get("type", "text"),
                    "page": child.get("page", 0),
                    "section": child.get("section", ""),
                    "language": child.get("language", "en"),
                    "word_count": len(child.get("content_raw", "").split()),
                    "children": [],
                })

    return tree


@router.get("/documents/{doc_id}/chunks")
async def get_document_chunks(
    doc_id: str,
    type_filter: str = Query(default=None, description="Filter by chunk type"),
    page_filter: int = Query(default=None, description="Filter by page number"),
    search: str = Query(default=None, description="Search chunk content"),
    user: dict = Depends(get_current_user),
):
    # Verify document ownership
    async with AsyncSessionLocal() as session:
        doc_result = await session.execute(
            select(Document).where(
                Document.doc_id == doc_id,
                Document.user_id == user["user_id"],
            )
        )
        if not doc_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Document not found")

        # Fetch parents from PostgreSQL
        query = select(ParentChunk).where(ParentChunk.doc_id == doc_id)
        if type_filter:
            query = query.where(ParentChunk.type == type_filter)
        if page_filter is not None:
            query = query.where(ParentChunk.page == page_filter)

        result = await session.execute(query)
        parents = result.scalars().all()

    # Fetch children from Qdrant
    qdrant = QdrantWrapper()
    children = await qdrant.get_by_doc_id(doc_id)

    # Apply filters to children
    if type_filter:
        children = [c for c in children if c.get("type") == type_filter]
    if page_filter is not None:
        children = [c for c in children if c.get("page") == page_filter]
    if search:
        search_lower = search.lower()
        children = [
            c for c in children
            if search_lower in c.get("content_raw", "").lower()
            or search_lower in c.get("content", "").lower()
        ]
        parents = [
            p for p in parents
            if search_lower in (p.content_raw or "").lower()
        ]

    tree = _build_chunk_tree(parents, children)
    return {"doc_id": doc_id, "chunks": tree, "total": len(tree)}
```

**Step 4: Add `get_by_doc_id` to QdrantWrapper**

In `backend/app/vectorstore/qdrant_client.py`, add method:

```python
async def get_by_doc_id(self, doc_id: str) -> list[dict]:
    """Fetch all chunks for a document (for ChunkViewer)."""
    try:
        results = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        return [point.payload for point in results[0]]
    except Exception as e:
        logger.warning("qdrant_scroll_failed", error=str(e))
        return []
```

**Step 5: Register router in main.py**

Add import and router:

```python
from backend.app.api import auth, chat, chunks, documents, eval, health

app.include_router(chunks.router, prefix="/api/v1", tags=["chunks"])
```

**Step 6: Run tests**

Run: `pytest tests/unit/test_chunks_api.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add backend/app/api/chunks.py backend/app/vectorstore/qdrant_client.py backend/app/main.py tests/unit/test_chunks_api.py
git commit -m "feat: chunks API — parent-child tree with filtering and search"
```

---

### Task 4.2: ChunkViewer Frontend — Store + API Client

**Files:**
- Create: `frontend/src/api/chunks.ts`
- Create: `frontend/src/stores/chunkStore.ts`
- Create: `frontend/src/hooks/useChunks.ts`

**Step 1: Create API client**

```typescript
// frontend/src/api/chunks.ts
import api from "./client";

export interface ChunkChild {
  chunk_id: string;
  parent_id: string;
  content: string;
  content_raw: string;
  type: string;
  page: number;
  section: string;
}

export interface ChunkNode {
  chunk_id: string;
  content_raw: string;
  content_markdown: string | null;
  content_html: string | null;
  type: string;
  page: number;
  section: string;
  language: string;
  word_count: number;
  children: ChunkChild[];
}

export interface ChunksResponse {
  doc_id: string;
  chunks: ChunkNode[];
  total: number;
}

export async function fetchDocumentChunks(
  docId: string,
  filters?: { type?: string; page?: number; search?: string },
): Promise<ChunksResponse> {
  const params = new URLSearchParams();
  if (filters?.type) params.set("type_filter", filters.type);
  if (filters?.page !== undefined) params.set("page_filter", String(filters.page));
  if (filters?.search) params.set("search", filters.search);

  const { data } = await api.get(`/documents/${docId}/chunks?${params}`);
  return data;
}
```

**Step 2: Create store**

```typescript
// frontend/src/stores/chunkStore.ts
import { create } from "zustand";
import { type ChunkNode, fetchDocumentChunks } from "@/api/chunks";

interface ChunkState {
  chunks: ChunkNode[];
  selectedChunk: ChunkNode | null;
  isLoading: boolean;
  filters: { type?: string; page?: number; search?: string };

  loadChunks: (docId: string) => Promise<void>;
  selectChunk: (chunk: ChunkNode | null) => void;
  setFilters: (filters: Partial<ChunkState["filters"]>) => void;
}

export const useChunkStore = create<ChunkState>((set, get) => ({
  chunks: [],
  selectedChunk: null,
  isLoading: false,
  filters: {},

  loadChunks: async (docId) => {
    set({ isLoading: true });
    try {
      const data = await fetchDocumentChunks(docId, get().filters);
      set({ chunks: data.chunks, isLoading: false });
    } catch {
      set({ chunks: [], isLoading: false });
    }
  },

  selectChunk: (chunk) => set({ selectedChunk: chunk }),

  setFilters: (filters) =>
    set((s) => ({ filters: { ...s.filters, ...filters } })),
}));
```

**Step 3: Create hook**

```typescript
// frontend/src/hooks/useChunks.ts
import { useEffect } from "react";
import { useChunkStore } from "@/stores/chunkStore";

export function useChunks(docId: string | null) {
  const { chunks, isLoading, loadChunks, filters } = useChunkStore();

  useEffect(() => {
    if (docId) loadChunks(docId);
  }, [docId, filters, loadChunks]);

  return { chunks, isLoading };
}
```

**Step 4: Commit**

```bash
git add frontend/src/api/chunks.ts frontend/src/stores/chunkStore.ts frontend/src/hooks/useChunks.ts
git commit -m "feat: ChunkViewer store, API client, and hook"
```

---

### Task 4.3: ChunkViewer Frontend — Page + Components

**Files:**
- Create: `frontend/src/pages/Chunks.tsx`
- Create: `frontend/src/components/chunks/ChunkTree.tsx`
- Create: `frontend/src/components/chunks/ChunkDetail.tsx`
- Create: `frontend/src/components/chunks/ChunkFilters.tsx`
- Modify: `frontend/src/App.tsx` — add Chunks page to nav

**Step 1: Create ChunkFilters component**

A horizontal bar with:
- Type dropdown (All, Text, Table, Figure, Code)
- Page number input
- Search text input
- Calls `useChunkStore().setFilters()`

**Step 2: Create ChunkTree component**

A list/tree view:
- Each parent chunk as an expandable accordion row
- Shows: type badge, section name, page number, word count
- Expanding reveals children with their enriched content preview
- Clicking a parent or child calls `selectChunk()`
- Color-code by type (text=default, table=blue, figure=purple, code=green)

**Step 3: Create ChunkDetail component**

A detail panel (right side) showing the selected chunk:
- Tabs: Raw | Enriched | Markdown | HTML
- Metadata section: type, page, section, language, word count
- For tables: render HTML content
- For figures: show image if available + description
- Children count badge

**Step 4: Create Chunks page**

Layout: document selector dropdown at top, then split view — ChunkTree on left (40%), ChunkDetail on right (60%). ChunkFilters bar between selector and split view.

Use the Glass Observatory styling to match existing pages.

**Step 5: Update App.tsx**

Add `"chunks"` to the `Page` type. Add nav pill with `Layers` icon from lucide-react. Render `<Chunks />` when active.

```typescript
type Page = "chat" | "documents" | "chunks";

// In nav pills array:
{ key: "chunks", label: "Chunks", icon: Layers },
```

**Step 6: Commit**

```bash
git add frontend/src/pages/Chunks.tsx frontend/src/components/chunks/ frontend/src/App.tsx
git commit -m "feat: ChunkViewer page — tree view, detail panel, filters"
```

---

## Summary

| Phase | Tasks | Key Deliverables |
|-------|-------|-----------------|
| 1. CohereReranker | 1.1–1.3 | Config, implementation with fallback, factory wiring |
| 2. JWT Auth | 2.1–2.7 | User model, auth service, protected endpoints, login page |
| 3. Prometheus + Grafana | 3.1–3.5 | Metrics definitions, middleware, LLM/pipeline instrumentation, Docker services |
| 4. ChunkViewer | 4.1–4.3 | Chunks API, frontend store, tree view page |

Total: **17 tasks**, each independently committable.
