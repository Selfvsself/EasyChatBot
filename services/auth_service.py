from datetime import datetime, timedelta

from core.config import settings
from core.jwt_handler import create_token
from core.security import hash_password, verify_password
from repositories.user_repository import UserRepository


class AuthService:

    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo

    def register(self, username: str, password: str):
        existing = self.user_repo.get_by_username_case_insensitive(username)

        if existing:
            raise Exception("User already exists")

        password_hash = hash_password(password)

        user = self.user_repo.create(username, password_hash)

        return user

    def login(self, username: str, password: str):
        user = self.user_repo.get_by_username_case_insensitive(username)

        if not user:
            raise Exception("Invalid credentials")

        now = datetime.utcnow()

        if user.locked_until and user.locked_until > now:
            raise Exception("Invalid credentials")

        if not verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= settings.LOGIN_MAX_FAILED_ATTEMPTS:
                user.locked_until = now + timedelta(minutes=settings.LOGIN_LOCKOUT_MINUTES)
                user.failed_login_attempts = 0
            self.user_repo.db.commit()
            raise Exception("Invalid credentials")

        user.failed_login_attempts = 0
        user.locked_until = None
        self.user_repo.db.commit()

        token = create_token(user.username)

        return token
