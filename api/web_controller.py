from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi import Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from core.security import get_current_user_from_cookie
from database import SessionLocal
from repositories.app_repository import AppRepository
from repositories.chat_repository import ChatRepository
from repositories.message_repository import MessageRepository
from repositories.user_repository import UserRepository
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


def get_app_repo(db: Session = Depends(get_db)):
    return AppRepository(db)


@router.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse(
        request,
        "login.html",
        {
        })


@router.get("/apps")
def apps_page(request: Request):
    return templates.TemplateResponse(
        request,
        "apps.html",
        {
        })


@router.get("/apps/{app_id}/chat")
def app_chats_page(app_id: UUID,
                   request: Request,
                   app_repo: AppRepository = Depends(get_app_repo),
                   chat_repo: ChatRepository = Depends(get_chat_repo),
                   user_repo: UserRepository = Depends(get_user_repo),
                   user_name: str = Depends(get_current_user_from_cookie)):
    app = app_repo.get_by_id(app_id)

    if app.chat_mode == 'STATELESS' or app.chat_mode == 'SINGLE_CHAT':
        user = user_repo.get_by_username(user_name)
        service = ChatService(
            chat_repo=chat_repo
        )
        chats = service.get_chats_by_user_and_app(app_id, user.id, 1, 0)
        if chats:
            chat_id = chats[0].id
        else:
            new_chat = service.create_chat(app_id=app_id, user_id=user.id, title=app.name)
            chat_id = new_chat.id
        target_url = f"/web/apps/{app_id}/chat/{chat_id}"
        return RedirectResponse(url=target_url)

    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "app_name": app.name,
            "app_id": app_id,
        })


@router.get("/apps/{app_id}/chat/{chat_id}")
def chat_page(app_id: UUID,
              chat_id: UUID,
              request: Request,
              app_repo: AppRepository = Depends(get_app_repo),
              chat_repo: ChatRepository = Depends(get_chat_repo)):
    app = app_repo.get_by_id(app_id)
    chat = chat_repo.get_by_id(chat_id)
    return templates.TemplateResponse(
        request,
        "message.html",
        {
            "app_id": app_id,
            "app_chat_mode": app.chat_mode,
            "chat_name": chat.title,
            "chat_id": chat_id
        }
    )
