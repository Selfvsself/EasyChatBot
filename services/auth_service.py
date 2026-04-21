from core.jwt_handler import create_token
from core.security import hash_password, verify_password
from repositories.user_repository import UserRepository


class AuthService:

    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo

    def register(self, username: str, password: str):
        existing = self.user_repo.get_by_username(username)

        if existing:
            raise Exception("User already exists")

        password_hash = hash_password(password)

        user = self.user_repo.create(username, password_hash)

        return user

    def login(self, username: str, password: str):
        user = self.user_repo.get_by_username(username)

        if not user:
            raise Exception("Invalid credentials")

        if not verify_password(password, user.password_hash):
            raise Exception(f"Invalid credentials")

        token = create_token(user.username)

        return token
