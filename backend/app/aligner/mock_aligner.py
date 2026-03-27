import asyncio

class MockAligner:
    def __init__(self):
        self.word_index = 0
        self.running = False

    async def next_position(self):
        self.running = True
        while self.running:
            await asyncio.sleep(1.0)
            yield {
                "type": "scroll",
                "data": {
                    "current_line_index": self.word_index // 5,
                    "current_word_index": self.word_index,
                    "confidence": 0.95,
                }
            }
            self.word_index += 1
        
    def stop(self):
        self.running = False
