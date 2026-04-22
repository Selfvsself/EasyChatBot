import uuid

from models.app_tool import AppTool
from models.tool import Tool
from repositories.base import BaseRepository


class AppToolRepository(BaseRepository):

    def get_enabled_tool_codes_by_app(self, app_id: uuid.UUID) -> list[str]:
        rows = (
            self.db.query(Tool.code)
            .join(AppTool, AppTool.tool_id == Tool.id)
            .filter(
                AppTool.app_id == app_id,
                AppTool.is_enabled == True,
                Tool.is_enabled == True
            )
            .all()
        )
        return [code for (code,) in rows]

    def replace_tools_for_app(self, app_id: uuid.UUID, tool_ids: list[uuid.UUID]):
        (
            self.db.query(AppTool)
            .filter(AppTool.app_id == app_id)
            .delete(synchronize_session=False)
        )

        links = [
            AppTool(app_id=app_id, tool_id=tool_id, is_enabled=True)
            for tool_id in tool_ids
        ]
        if links:
            self.db.add_all(links)

        self.db.commit()
