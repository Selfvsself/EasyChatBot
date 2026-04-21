from models.message import Message
from repositories.base import BaseRepository


class MessageRepository(BaseRepository):

    def create_msg(
            self,
            chat_id,
            role,
            text,
            status="sent"
    ):
        msg = Message(
            chat_id=chat_id,
            role=role,
            text=text,
            status=status
        )
        return self.add(msg)

    def get_by_chat(self, chat_id, limit=50, offset=0):
        return (
            self.db.query(Message)
            .filter(Message.chat_id == chat_id)
            .order_by(Message.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def update_status(self, message_id, status):
        msg = self.db.get(Message, message_id)
        if msg:
            msg.status = status
            self.db.commit()
        return msg

    def mark_as_delivered(self, message_id):
        return self.update_status(message_id, "delivered")

    def mark_as_read(self, message_id):
        return self.update_status(message_id, "read")
