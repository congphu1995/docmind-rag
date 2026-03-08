# tests/unit/test_auth_api.py
import pytest

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
