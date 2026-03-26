import asyncio
from app.schemas import ScrollMessage, StatusMessage

class MockAligner:
    def __init__(self):
        self.word_index = 0
        self.running = False

    async def next_position(self):
        self.running = True
        while self.running:
            await asyncio.sleep(1.0)
            self.word_index += 1
            yield ScrollMessage(
                type='scroll',
                word_index=self.word_index,
                line_index=self.word_index // 5,
                confidence=0.9
            )
        
    def stop(self):
        self.running = False
        return StatusMessage(type='status', status='stopped', detail="Alignment stopped")