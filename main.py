import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.apps_controller import router as apps_router
from api.auth_controller import router as auth_router
from api.chat_controller import router as chat_router
from api.message_controller import router as msg_router
from api.web_controller import router as web_router
from core.config import settings
from database import SessionLocal
from services.kafka_service import KafkaService
from workers.kafka_consumer import consume_and_dispatch
from workers.msg_worker import run_worker
from ws.connection_manager import ConnectionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager = ConnectionManager()
    logging.info("Web socket connections created")
    kafka = KafkaService(
        settings.KAFKA_BOOTSTRAP_SERVERS,
        settings.KAFKA_REQUEST_TOPIC,
        settings.KAFKA_RESPONSE_TOPIC)

    await kafka.start()
    logging.info("Kafka connections created")
    app.state.kafka = kafka
    app.state.manager = manager

    task = asyncio.create_task(
        consume_and_dispatch(kafka, manager, SessionLocal)
    )
    logging.info("Consumers for websocket started")

    worker_task = asyncio.create_task(run_worker())
    logging.info("Message worker started")

    logging.info("Server started ...")
    yield
    task.cancel()
    worker_task.cancel()
    await kafka.stop()
    logging.info("Server stopped.")


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(web_router, prefix="/web")
app.include_router(apps_router, prefix="/api/apps")
app.include_router(chat_router, prefix="/api/apps")
app.include_router(msg_router, prefix="/api/apps")
app.include_router(auth_router, prefix="/api/auth")


@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/web/login")


if __name__ == "__main__":
    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT, log_config=None)
