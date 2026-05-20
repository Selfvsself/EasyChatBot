from models.chat import Chat
from repositories.base import BaseRepository
from sqlalchemy import func


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
            .order_by(func.coalesce(Chat.updated_at, Chat.created_at).desc(), Chat.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def update_memory_summary(self, chat_id, memory_summary: str):
        chat = self.db.get(Chat, chat_id)
        if chat:
            chat.memory_summary = memory_summary
            self.db.commit()
            self.db.refresh(chat)
        return chat

    def update_title(self, chat_id, title: str):
        chat = self.db.get(Chat, chat_id)
        if chat:
            chat.title = title
            self.db.commit()
            self.db.refresh(chat)
        return chat

    def touch(self, chat_id):
        chat = self.db.get(Chat, chat_id)
        if chat:
            chat.updated_at = func.now()
            self.db.commit()
            self.db.refresh(chat)
        return chat
