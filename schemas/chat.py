from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class ChatRequest(BaseModel):
    title: str | None = None


class ChatResponse(BaseModel):
    id: UUID
    user_id: UUID
    title: str | None = None
    memory_summary: str | None = None


class ChatItem(BaseModel):
    id: UUID
    user_id: UUID
    title: str | None = None
    memory_summary: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True  # 🔥 важно для SQLAlchemy

class ChatListResponse(BaseModel):
    chats: list[ChatItem]
