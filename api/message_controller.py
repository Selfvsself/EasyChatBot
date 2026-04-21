from uuid import UUID

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi import Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from core.security import get_current_user_ws, get_current_user_http
from database import SessionLocal
from repositories.chat_repository import ChatRepository
from repositories.message_repository import MessageRepository
from repositories.user_repository import UserRepository
from schemas.message import ChatHistoryResponse
from schemas.message import MessageRequest
from services.chat_service import ChatService
from services.message_service import MessageService

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_msg_repo(db: Session = Depends(get_db)):
    return MessageRepository(db)


def get_chat_repo(db: Session = Depends(get_db)):
    return ChatRepository(db)


def get_user_repo(db: Session = Depends(get_db)):
    return UserRepository(db)


@router.websocket("/{app_id}/chat/{chat_id}/ws")
async def websocket_endpoint(app_id: str,
                             chat_id: str,
                             ws: WebSocket,
                             chat_repo: ChatRepository = Depends(get_chat_repo),
                             user_repo: UserRepository = Depends(get_user_repo),
                             ):
    manager = ws.app.state.manager
    user_id = await get_current_user_ws(ws)
    if not user_id:
        return
    user = user_repo.get_by_username(user_id)
    service = ChatService(
        chat_repo=chat_repo
    )
    chat = service.get_chat_by_user_and_app(chat_id=UUID(chat_id), app_id=UUID(app_id), user_id=user.id)

    if not chat:
        return

    await manager.connect(str(chat_id), ws)

    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(str(chat_id), ws)


@router.post("/{app_id}/chat/{chat_id}/send")
async def send_message(app_id: UUID,
                       chat_id: UUID,
                       payload: MessageRequest,
                       request: Request,
                       message_repo: MessageRepository = Depends(get_msg_repo),
                       chat_repo: ChatRepository = Depends(get_chat_repo),
                       user_repo: UserRepository = Depends(get_user_repo),
                       user_id: str = Depends(get_current_user_http)):
    user = user_repo.get_by_username(user_id)
    service = ChatService(
        chat_repo=chat_repo
    )
    chat = service.get_chat_by_user_and_app(chat_id=chat_id, app_id=app_id, user_id=user.id)

    if chat:
        kafka = request.app.state.kafka
        service = MessageService(
            kafka_service=kafka,
            message_repo=message_repo
        )
        return await service.send_message(chat_id, user.id, payload.text)
    else:
        raise HTTPException(status_code=403, detail="Forbidden")


@router.get("/{app_id}/chat/{chat_id}/history", response_model=ChatHistoryResponse)
def get_chat_history(
        app_id: UUID,
        chat_id: UUID,
        limit: int = Query(50, le=100),
        offset: int = Query(0),
        request: Request = None,
        message_repo: MessageRepository = Depends(get_msg_repo),
        chat_repo: ChatRepository = Depends(get_chat_repo),
        user_repo: UserRepository = Depends(get_user_repo),
        user_id: str = Depends(get_current_user_http)
):
    user = user_repo.get_by_username(user_id)
    service = ChatService(
        chat_repo=chat_repo
    )
    chat = service.get_chat_by_user_and_app(chat_id=chat_id, app_id=app_id, user_id=user.id)

    if chat:
        kafka = request.app.state.kafka

        service = MessageService(
            kafka_service=kafka,
            message_repo=message_repo
        )

        messages = service.get_history(chat_id, limit, offset)

        return {"messages": messages}
    else:
        raise HTTPException(status_code=403, detail="Forbidden")
