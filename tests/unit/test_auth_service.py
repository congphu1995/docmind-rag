import pytest

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
    import jwt as pyjwt
    from datetime import datetime, timedelta
    from backend.app.core.config import settings

    payload = {
        "user_id": "user-1",
        "type": "access",
        "exp": datetime.utcnow() - timedelta(hours=1),
    }
    token = pyjwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    result = auth_service.decode_token(token)
    assert result is None
