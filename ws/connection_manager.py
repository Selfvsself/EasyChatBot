from fastapi import WebSocket
import logging


class ConnectionManager:
    def __init__(self):
        self.connections: dict[str, list[WebSocket]] = {}

    async def connect(self, user_id: str, ws: WebSocket):
        await ws.accept()
        self.connections.setdefault(user_id, []).append(ws)
        logging.info("User '%s' connected", user_id)

    def disconnect(self, user_id: str, ws: WebSocket):
        self.connections[user_id].remove(ws)
        logging.info("User '%s' has been disconnect", user_id)

    async def send_to_user(self, user_id: str, message: dict):
        if user_id in self.connections:
            for ws in self.connections[user_id]:
                await ws.send_json(message)
