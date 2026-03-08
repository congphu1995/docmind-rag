# backend/app/api/auth.py
from fastapi import APIRouter, HTTPException

from backend.app.schemas.auth import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    TokenUser,
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
        user=TokenUser(email=user.email, username=user.username),
    )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    user = await _auth.authenticate(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return TokenResponse(
        access_token=_auth.create_access_token(user.id, user.email),
        refresh_token=_auth.create_refresh_token(user.id),
        user=TokenUser(email=user.email, username=user.username),
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
        user=TokenUser(email=user.email, username=user.username),
    )
