from models.chat import Chat
from repositories.base import BaseRepository


class ChatRepository(BaseRepository):

    def get_by_id(self, chat_id):
        chat = self.db.get(Chat, chat_id)
        return chat

    def create_chat(self, user_id, app_id, title=None):
        chat = Chat(
            user_id=user_id,
            app_id=app_id,
            title=title
        )
        return self.add(chat)

    def get_by_user_and_app(self, user_id, app_id, limit=50, offset=0):
        return (
            self.db.query(Chat)
            .filter(
                Chat.user_id == user_id,
                Chat.app_id == app_id
            )
            .order_by(Chat.updated_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )