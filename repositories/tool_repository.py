from models.tool import Tool
from repositories.base import BaseRepository


class ToolRepository(BaseRepository):

    def get_all_enabled(self):
        return (
            self.db.query(Tool)
            .filter(Tool.is_enabled == True)
            .all()
        )

    def get_all(self):
        return self.db.query(Tool).all()

    def get_by_codes(self, codes: list[str]):
        if not codes:
            return []

        return (
            self.db.query(Tool)
            .filter(Tool.code.in_(codes))
            .all()
        )

    def ensure_defaults(self, defaults: list[dict]):
        if not defaults:
            return

        existing = {code for (code,) in self.db.query(Tool.code).all()}
        to_create = [
            Tool(
                code=item["code"],
                name=item["name"],
                description=item.get("description"),
                is_enabled=item.get("is_enabled", True)
            )
            for item in defaults
            if item["code"] not in existing
        ]

        if not to_create:
            return

        self.db.add_all(to_create)
        self.db.commit()
