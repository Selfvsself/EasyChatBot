from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from core.security import get_current_user_http
from database import SessionLocal
from repositories.chat_repository import ChatRepository
from repositories.message_repository import MessageRepository
from repositories.user_repository import UserRepository
from schemas.chat import ChatListResponse, ChatItem, ChatRequest
from services.chat_service import ChatService

router = APIRouter()

templates = Jinja2Templates(directory="static")


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


@router.get("/{app_id}/chat", response_model=ChatListResponse)
def get_all_chats(
        app_id: UUID,
        limit: int = Query(50, le=100),
        offset: int = Query(0),
        chat_repo: ChatRepository = Depends(get_chat_repo),
        user_repo: UserRepository = Depends(get_user_repo),
        user_name: str = Depends(get_current_user_http)
):
    service = ChatService(
        chat_repo=chat_repo
    )

    user = user_repo.get_by_username(user_name)

    chats = service.get_chats_by_user_and_app(app_id, user.id, limit, offset)

    return {"chats": chats}


@router.post("/{app_id}/chat", response_model=ChatItem)
def create_chat(
        app_id: UUID,
        payload: ChatRequest,
        chat_repo: ChatRepository = Depends(get_chat_repo),
        user_repo: UserRepository = Depends(get_user_repo),
        user_name: str = Depends(get_current_user_http)
):
    service = ChatService(
        chat_repo=chat_repo
    )

    user = user_repo.get_by_username(user_name)

    chat = service.create_chat(app_id, user.id, payload.title)

    return chat
