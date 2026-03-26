from typing import Literal
from pydantic import BaseModel

class StatusMessage(BaseModel):
    type: Literal['status']
    status: str
    detail: str | None = None

class ScrollMessage(BaseModel):
    type: Literal['scroll']
    word_index: int
    line_index: int
    confidence: float

class ErrorMessage(BaseModel):
    type: Literal['error']
    msg: str | None = None

class ClientCommand(BaseModel):
    type: Literal['start', 'stop', 'ping']