from uuid import UUID

from pydantic import BaseModel


class AppItem(BaseModel):
    id: UUID
    name: str
    description: str
    icon: str | None

    class Config:
        from_attributes = True  # 🔥 важно для SQLAlchemy


class AppListResponse(BaseModel):
    items: list[AppItem]
