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


def build_status_message(*, is_running: bool, confidence: float = 0.0) -> dict:
    return {
        "type": "status",
        "data": {
            "is_running": is_running,
            "current_line_index": 0,
            "current_word_index": 0,
            "confidence": confidence,
            "script_lines": settings.demo_script_lines,
        },
    }


def build_error_message(*, action: str | None) -> dict:
    allowed_actions = ["connect", "start", "stop", "disconnect"]
    return {
        "type": "error",
        "error": {
            "code": "invalid_action",
            "message": "Unknown action received.",
            "received_action": action,
            "allowed_actions": allowed_actions,
        },
    }


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
    await manager.connect(websocket)
    mock_aligner_task = None
    is_cleaned_up = False

    async def stop_alignment():
        nonlocal mock_aligner_task
        mock_aligner.stop()
        if mock_aligner_task and not mock_aligner_task.done():
            mock_aligner_task.cancel()
            try:
                await mock_aligner_task
            except asyncio.CancelledError:
                pass
        mock_aligner_task = None

    async def cleanup_connection():
        nonlocal is_cleaned_up
        if is_cleaned_up:
            return
        await stop_alignment()
        manager.disconnect(websocket)
        is_cleaned_up = True

    async def handle_action(action: str | None) -> bool:
        nonlocal mock_aligner_task

        if action == "connect":
            await manager.send_json(
                websocket=websocket,
                message=build_status_message(is_running=False),
            )
            return False

        if action == "start":
            if mock_aligner_task is None or mock_aligner_task.done():
                mock_aligner_task = asyncio.create_task(stream_mock_aligner(websocket))
            await manager.send_json(
                websocket=websocket,
                message=build_status_message(is_running=True),
            )
            return False

        if action == "stop":
            await stop_alignment()
            await manager.send_json(
                websocket=websocket,
                message=build_status_message(is_running=False),
            )
            return False

        if action == "disconnect":
            await cleanup_connection()
            await manager.send_json(
                websocket=websocket,
                message=build_status_message(is_running=False),
            )
            return True

        await manager.send_json(
            websocket=websocket,
            message=build_error_message(action=action),
        )
        return False

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            should_break = await handle_action(action)
            if should_break:
                break
    except WebSocketDisconnect:
        pass
    finally:
        await cleanup_connection()
