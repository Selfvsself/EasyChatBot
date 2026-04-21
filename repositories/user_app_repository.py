from models.user_app import UserApp
from repositories.base import BaseRepository


class UserAppRepository(BaseRepository):

    def add_access(self, user_id, app_id):
        entity = UserApp(user_id=user_id, app_id=app_id)
        return self.add(entity)

    def has_access(self, user_id, app_id) -> bool:
        return (
            self.db.query(UserApp)
            .filter(
                UserApp.user_id == user_id,
                UserApp.app_id == app_id
            )
            .first()
            is not None
        )

    def get_user_apps(self, user_id):
        return (
            self.db.query(UserApp)
            .filter(UserApp.user_id == user_id)
            .all()
        )