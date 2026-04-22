import json
from uuid import UUID
import logging
from core.config import settings
from repositories.message_repository import MessageRepository
from repositories.chat_repository import ChatRepository
from services.chat_memory_service import ChatMemoryService
from services.llm_client import LLMClient


async def consume_and_dispatch(kafka, manager, db_factory):
    llm_client = LLMClient(
        base_url=settings.LLM_URL,
        model=settings.LLM_MODEL
    )

    async for msg in kafka.consumer:
        data = json.loads(msg.value)
        chat_id = data.get("chat_id")
        user_id = data.get("user_id")
        role = data.get("role", "assistant")
        if not user_id:
            logging.warning("No user_id in message in chat %s", chat_id)
            continue
        text = data.get('text')
        event = data.get("event")

        db = db_factory()
        try:
            chat_repo = ChatRepository(db)
            chat = chat_repo.get_by_id(chat_id)
            if not chat:
                logging.warning("Chat %s not found while dispatching message", chat_id)
                continue

            if chat.user_id == UUID(user_id):
                if event == "agent_stage":
                    await manager.send_to_user(chat_id, data)
                    continue

                msg_repo = MessageRepository(db)

                added_msg = msg_repo.create_msg(
                    chat_id=chat_id,
                    role=role,
                    text=text
                )

                memory_service = ChatMemoryService(
                    message_repo=msg_repo,
                    chat_repo=chat_repo,
                    llm_client=llm_client
                )
                await memory_service.compress_if_needed(chat_id=chat_id)

                await manager.send_to_user(chat_id, data)
                logging.info("Message %s has been send to '%s' chat", added_msg.id, chat_id)
        except Exception as db_error:
            logging.error(f"❌ DB error: {db_error}")
        finally:
            db.close()
