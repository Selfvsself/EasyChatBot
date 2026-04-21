import logging

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from core.security import get_current_user_http
from database import SessionLocal
from repositories.user_repository import UserRepository
from schemas.auth import RegisterRequest, LoginRequest, TokenResponse
from services.auth_service import AuthService

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_user_repo(db: Session = Depends(get_db)):
    return UserRepository(db)


def get_auth_service(repo: UserRepository = Depends(get_user_repo)):
    return AuthService(repo)


@router.post("/register")
def register(data: RegisterRequest, service: AuthService = Depends(get_auth_service)):
    raise HTTPException(status_code=403, detail="Forbidden")
    try:
        service.register(data.username, data.password)
        return {"message": "User created"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=TokenResponse)
def login(response: Response, data: LoginRequest, service: AuthService = Depends(get_auth_service)):
    username = data.username
    try:
        logging.info("User '%s' trying to login", username)
        token = service.login(username, data.password)
        logging.info("User '%s' has been login successfully", username)
        response.set_cookie(key="access_token", value=token, httponly=True)
        return {"access_token": token, "username": username}
    except Exception:
        logging.info("User '%s' not logged by 'Invalid credentials' error", username)
        raise HTTPException(status_code=401, detail="Invalid credentials")


@router.post("/logout")
def login(response: Response,
          user_repo: UserRepository = Depends(get_user_repo),
          user_name: str = Depends(get_current_user_http)):
    user = user_repo.get_by_username(user_name)
    if user:
        response.delete_cookie(
            key="access_token",
            path="/web/login",
            httponly=True,
            samesite="lax"
        )
        return {"status": "ok", "message": "Logged out successfully"}
    raise HTTPException(status_code=401, detail="Invalid credentials")
