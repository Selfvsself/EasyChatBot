import uuid


class ApplicationService:

    def __init__(self, apps_repo, user_apps_repo):
        self.apps_repo = apps_repo
        self.user_apps_repo = user_apps_repo

    def get_apps_by_userid(self, user_id: uuid):
        user_apps = self.user_apps_repo.get_user_apps(user_id=user_id)
        user_app_ids = {app.app_id for app in user_apps}

        visible_apps = self.apps_repo.get_all_visible()

        return [app for app in visible_apps if app.id in user_app_ids]


