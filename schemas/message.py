from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class MessageRequest(BaseModel):
    text: str


class ChatResponse(BaseModel):
    msg_id: str
    text: str


class ChatHistoryItem(BaseModel):
    id: UUID
    text: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True  # 🔥 важно для SQLAlchemy


class ChatHistoryResponse(BaseModel):
    messages: list[ChatHistoryItem]


class ChatItem(BaseModel):
    id: UUID
    user_id: int
    title: str
    created_at: datetime

    class Config:
        from_attributes = True  # 🔥 важно для SQLAlchemy

class ChatListResponse(BaseModel):
    chats: list[ChatItem]
