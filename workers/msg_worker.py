import json
import logging

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from sqlalchemy.orm import Session

from core.config import settings
from database import SessionLocal
from repositories.app_repository import AppRepository
from repositories.chat_repository import ChatRepository
from repositories.message_repository import MessageRepository
from services.llm_client import LLMClient
from services.message_processor import MessageProcessor

logger = logging.getLogger(__name__)


async def run_worker():
    consumer = AIOKafkaConsumer(
        settings.KAFKA_REQUEST_TOPIC,
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id="worker-group"
    )

    producer = AIOKafkaProducer(
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS
    )

    await consumer.start()
    await producer.start()

    logger.info("✅ Worker started")

    try:
        async for msg in consumer:
            data = json.loads(msg.value)

            chat_id = data["chat_id"]
            user_id = data["user_id"]
            role = data["role"]
            text = data["text"]

            logger.info(f"📥 Received message from user: {user_id}")

            db: Session = SessionLocal()

            try:
                # =========================
                # REPOSITORIES
                # =========================
                message_repo = MessageRepository(db)
                chat_repo = ChatRepository(db)
                app_repo = AppRepository(db)
                llm_client = LLMClient(
                    base_url=settings.LLM_URL,
                    model=settings.LLM_MODEL
                )

                # =========================
                # PROCESSOR
                # =========================
                processor = MessageProcessor(
                    message_repo=message_repo,
                    chat_repo=chat_repo,
                    app_repo=app_repo,
                    llm_client=llm_client
                )

                response_text = await processor.process(
                    chat_id=chat_id,
                    user_id=user_id,
                    text=text
                )

                # =========================
                # SEND RESPONSE
                # =========================
                response = {
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "role": "assistant",
                    "text": response_text
                }

                await producer.send_and_wait(
                    settings.KAFKA_RESPONSE_TOPIC,
                    json.dumps(response).encode()
                )

                logger.info(f"📤 Sent response for chat {chat_id}")

            except Exception as e:
                logger.exception("❌ Worker error")

            finally:
                db.close()

    finally:
        await consumer.stop()
        await producer.stop()
