import uuid


class ChatService:

    def __init__(self, chat_repo):
        self.chat_repo = chat_repo

    def create_chat(self, app_id, user_id: uuid, title: str):
        chat = self.chat_repo.create_chat(
            user_id=user_id,
            app_id=app_id,
            title=title
        )

        return chat

    def get_chats_by_user_and_app(self, app_id:uuid, user_id: uuid, limit: int, offset: int):
        messages = self.chat_repo.get_by_user_and_app(
            user_id=user_id,
            app_id=app_id,
            limit=limit,
            offset=offset
        )

        return list(messages)

    def get_chat_by_user_and_app(self, chat_id:uuid, app_id:uuid, user_id: uuid):
        chat = self.chat_repo.get_by_id(chat_id)

        if chat.user_id == user_id:
            if chat.app_id == app_id:
                return chat

        return None
