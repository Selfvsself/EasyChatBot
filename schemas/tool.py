from uuid import UUID

from pydantic import BaseModel


class ToolItem(BaseModel):
    id: UUID
    code: str
    name: str
    description: str | None = None

    class Config:
        from_attributes = True


class AppToolsMatrixRow(BaseModel):
    app_id: UUID
    app_name: str
    enabled_tool_codes: list[str]


class AppToolsMatrixResponse(BaseModel):
    tools: list[ToolItem]
    rows: list[AppToolsMatrixRow]


class UpdateAppToolsRequest(BaseModel):
    tool_codes: list[str]
