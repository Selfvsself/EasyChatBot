from models.message import Message
from repositories.base import BaseRepository


class MessageRepository(BaseRepository):

    def create_msg(
            self,
            chat_id,
            role,
            text,
            status=None
    ):
        if status is None:
            if role == "user":
                status = "processing"
            elif role == "assistant":
                status = "completed"
            else:
                status = "sent"

        msg = Message(
            chat_id=chat_id,
            role=role,
            text=text,
            status=status
        )
        return self.add(msg)

    def get_by_chat(self, chat_id, limit=50, offset=0, include_archived=True, reverse=False):
        query = self.db.query(Message).filter(Message.chat_id == chat_id)
        if not include_archived:
            query = query.filter(Message.is_archived.is_(False))

        return (
            query.order_by(Message.created_at.asc() if reverse else Message.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def count_active_by_chat(self, chat_id):
        return (
            self.db.query(Message)
            .filter(
                Message.chat_id == chat_id,
                Message.is_archived.is_(False),
            )
            .count()
        )

    def get_oldest_active_by_chat(self, chat_id, limit=20):
        return (
            self.db.query(Message)
            .filter(
                Message.chat_id == chat_id,
                Message.is_archived.is_(False),
            )
            .order_by(Message.created_at.asc())
            .limit(limit)
            .all()
        )

    def mark_as_archived(self, message_ids):
        if not message_ids:
            return 0

        affected = (
            self.db.query(Message)
            .filter(Message.id.in_(message_ids))
            .update({Message.is_archived: True}, synchronize_session=False)
        )
        self.db.commit()
        return affected

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

    def complete_latest_processing_user_message(self, chat_id):
        msg = (
            self.db.query(Message)
            .filter(
                Message.chat_id == chat_id,
                Message.role == "user",
                Message.status.in_(["processing", "sent"]),
            )
            .order_by(Message.created_at.desc())
            .first()
        )
        if msg:
            msg.status = "completed"
            self.db.commit()
        return msg
