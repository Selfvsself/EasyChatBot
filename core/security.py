import jwt
from fastapi import WebSocket, Cookie, HTTPException, Header
from passlib.context import CryptContext

from core.config import settings

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


def hash_password(password: str):
    return pwd_context.hash(password)


def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)


def verify_token(token: str):
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload["sub"]
    except:
        return None


def get_current_user_http(authorization: str = Header(...)):
    scheme, token = authorization.split()

    if scheme.lower() != "bearer":
        raise HTTPException(status_code=401)

    user = verify_token(token)
    if not user:
        raise HTTPException(status_code=401)

    return user


async def get_current_user_ws(ws: WebSocket):
    token = ws.query_params.get("token")
    user = verify_token(token)

    if not user:
        await ws.close(code=1008)
        return None

    return user


def get_current_user_from_cookie(access_token: str = Cookie(None)):
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = verify_token(access_token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    return user
