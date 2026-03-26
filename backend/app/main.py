from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from app.schemas import StatusMessage, ScrollMessage, ErrorMessage, ClientCommand
from app.config import settings
from app.websocket_manager import ConnectionManager
from app.aligner.mock_aligner import MockAligner
from fastapi.responses import HTMLResponse
import asyncio

app = FastAPI()
mock_aligner = MockAligner()
manager = ConnectionManager()
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://127.0.0.1:8000/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""

async def stream_mock_aligner(websocket: WebSocket):
    async for scroll_msg in mock_aligner.next_position():
        await websocket.send_json(scroll_msg.model_dump())


@app.get("/")
async def root():
    return HTMLResponse(content=html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stream_task = None
    connected_msg = StatusMessage(type='status', status='connected', detail="WebSocket connection established")
    print("WebSocket connection established")
    await manager.send_json(websocket=websocket, message=connected_msg.model_dump())
    try:
        while True:
            data = await websocket.receive_text()

            if data == "start":
                if stream_task is None or stream_task.done():
                    stream_task = asyncio.create_task(stream_mock_aligner(websocket))
            elif data == "stop":
                stop_msg = mock_aligner.stop()
                if stream_task is not None:
                    stream_task.cancel()

                await websocket.send_json(stop_msg.model_dump())
            else:
                await websocket.send_text(f"""Unknown command: {data}. Valid commands are 'next' and 'stop'.""")

    except WebSocketDisconnect:
        print("WebSocket connection closed")
        manager.disconnect(websocket)


