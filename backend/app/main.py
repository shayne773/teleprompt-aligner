from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
            var ws = new WebSocket("ws://127.0.0.1:8000/ws/teleprompter");
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
        await websocket.send_json(scroll_msg)


@app.get("/")
async def root():
    return HTMLResponse(content=html)

'''
response format:
{
  "type": "status",
  "data": {
    "is_running": true,
    "current_line_index": 1,
    "current_word_index": 5,
    "confidence": 0.95,
    "script_lines": ["line 1", "line 2"]
  }
}
'''
@app.websocket("/ws/teleprompter")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    mock_aligner_task = None
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            if action == "connect":
                connected_msg = {
                    "type": "status",
                    "data": {
                        "is_running": False,
                        "current_line_index": 0,
                        "current_word_index": 0,
                        "confidence": 0.0,
                        "script_lines": settings.demo_script_lines
                    }
                }
                await manager.send_json(websocket=websocket, message=connected_msg)

            elif action == "start":
                if mock_aligner_task is None or mock_aligner_task.done():
                    mock_aligner_task = asyncio.create_task(stream_mock_aligner(websocket))
                start_msg = {
                    "type": "status",
                    "data": {
                        "is_running": True,
                    }
                }
                print("Starting mock aligner...")
                await manager.send_json(websocket=websocket, message=start_msg)
            elif action == "stop":
                mock_aligner.stop()
                if mock_aligner_task and not mock_aligner_task.done():
                    mock_aligner_task.cancel()
                stop_msg = {
                    "type": "status",
                    "data": {
                        "is_running": False,
                    }
                }
                await manager.send_json(websocket=websocket, message=stop_msg)
            elif action == "disconnect":
                mock_aligner.stop()
                if mock_aligner_task and not mock_aligner_task.done():
                    mock_aligner_task.cancel()
                disconnect_msg = {
                    "type": "status",
                    "data": {
                        "is_running": False,
                    }
                }

                await manager.send_json(websocket=websocket, message=disconnect_msg)
                manager.disconnect(websocket)
                break

    except WebSocketDisconnect:
        if mock_aligner_task and not mock_aligner_task.done():
            mock_aligner_task.cancel()
        manager.disconnect(websocket)
