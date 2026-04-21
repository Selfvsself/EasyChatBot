import json
from uuid import UUID
import logging
from repositories.message_repository import MessageRepository
from repositories.chat_repository import ChatRepository


async def consume_and_dispatch(kafka, manager, db_factory):
    async for msg in kafka.consumer:
        data = json.loads(msg.value)
        chat_id = data.get("chat_id")
        user_id = data.get("user_id")
        role = data.get("role", "assistant")
        if not user_id:
            logging.warning("No user_id in message in chat %s", chat_id)
            continue
        text = data.get('text')

        db = db_factory()
        try:
            chat_repo = ChatRepository(db)
            chat = chat_repo.get_by_id(chat_id)
            if chat.user_id == UUID(user_id):
                msg_repo = MessageRepository(db)

                added_msg = msg_repo.create_msg(
                    chat_id=chat_id,
                    role=role,
                    text=text
                )

                await manager.send_to_user(chat_id, data)
                logging.info("Message %s has been send to '%s' chat", added_msg.id, chat_id)
        except Exception as db_error:
            logging.error(f"❌ DB error: {db_error}")
        finally:
            db.close()
