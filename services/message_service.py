import logging
import uuid


class MessageService:

    def __init__(self, kafka_service, message_repo):
        self.kafka = kafka_service
        self.message_repo = message_repo

    async def send_message(self, chat_id: uuid, user_id:uuid, text: str):
        msg_id = uuid.uuid4()

        await self.kafka.send_response_message(
            chat_id=str(chat_id),
            user_id=str(user_id),
            text=text,
            role="user"
        )

        logging.info("Received message '%s' from '%s'", msg_id, chat_id)

        await self.kafka.send_chat_message(
            chat_id=str(chat_id),
            user_id=str(user_id),
            text=text,
            role="user"
        )

        return {"status": "ok", "msg_id": msg_id}

    def get_history(self, chat_id: uuid, limit: int, offset: int):
        messages = self.message_repo.get_by_chat(
            chat_id=chat_id,
            limit=limit,
            offset=offset
        )

        # 👉 переворачиваем, чтобы старые были сверху
        return list(reversed(messages))
