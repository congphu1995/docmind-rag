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
