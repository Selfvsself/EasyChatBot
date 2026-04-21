from sqlalchemy.orm import Session
from models.user import User
from repositories.base import BaseRepository


class UserRepository(BaseRepository):

    def get_by_id(self, user_id):
        return self.db.get(User, user_id)

    def get_by_username(self, username: str):
        return (
            self.db.query(User)
            .filter(User.username == username)
            .first()
        )

    def create(self, username: str, password_hash: str):
        user = User(
            username=username,
            password_hash=password_hash
        )
        return self.add(user)