from fastapi import APIRouter, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from core.security import get_current_user_http
from database import SessionLocal
from repositories.app_repository import AppRepository
from repositories.user_app_repository import UserAppRepository
from repositories.user_repository import UserRepository
from schemas.app import AppListResponse
from services.apps_service import ApplicationService

router = APIRouter()

templates = Jinja2Templates(directory="static")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_app_repo(db: Session = Depends(get_db)):
    return AppRepository(db)


def get_user_app_repo(db: Session = Depends(get_db)):
    return UserAppRepository(db)

def get_user_repo(db: Session = Depends(get_db)):
    return UserRepository(db)


@router.get("/", response_model=AppListResponse)
def get_all_apps(
        app_repo: AppRepository = Depends(get_app_repo),
        user_apps_repo: UserAppRepository = Depends(get_user_app_repo),
        user_repo: UserRepository = Depends(get_user_repo),
        user_name: str = Depends(get_current_user_http)
):
    service = ApplicationService(
        apps_repo=app_repo,
        user_apps_repo=user_apps_repo
    )

    user = user_repo.get_by_username(user_name)

    chats = service.get_apps_by_userid(user.id)

    return {"items": chats}
