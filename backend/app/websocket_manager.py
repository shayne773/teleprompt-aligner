from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        if websocket not in self.active_connections:
            self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_json(self, websocket: WebSocket, message: dict):
        await websocket.send_json(message)
    
    async def send_text(self, websocket: WebSocket, message: str):
        await websocket.send_text(message)
    
    #optional for current project scope but useful for future broadcast features
    async def broadcast_json(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)
    #optional for current project scope but useful for future broadcast features
    async def broadcast_text(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)
