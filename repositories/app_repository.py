import uuid

from models.app import App
from repositories.base import BaseRepository


class AppRepository(BaseRepository):

    def get_by_code(self, code: str):
        return (
            self.db.query(App)
            .filter(App.code == code)
            .first()
        )

    def get_by_id(self, id: uuid):
        return (
            self.db.query(App)
            .filter(App.id == id)
            .first()
        )

    def get_all_visible(self):
        return (
            self.db.query(App)
            .filter(App.is_visible == True)
            .all()
        )
