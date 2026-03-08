from datetime import datetime, timedelta

import bcrypt
import jwt
from sqlalchemy import select

from backend.app.core.config import settings
from backend.app.core.database import AsyncSessionLocal
from backend.app.core.logging import logger
from backend.app.models.user import User


class AuthService:

    def hash_password(self, password: str) -> str:
        return bcrypt.hashpw(
            password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

    def verify_password(self, plain: str, hashed: str) -> bool:
        return bcrypt.checkpw(
            plain.encode("utf-8"), hashed.encode("utf-8")
        )

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
