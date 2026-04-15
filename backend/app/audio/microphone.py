import logging

import pyaudiowpatch as pyaudio

logger = logging.getLogger(__name__)


class MicrophoneCapture:
    def __init__(
        self,
        channels: int = 1,
        rate: int = 16000,
        chunk: int = 256,
        seconds_per_window: int = 3,
        format_type=pyaudio.paInt16,
    ):
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.seconds_per_window = seconds_per_window
        self.format_type = format_type

        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []

        self.chunks_per_window = int(self.rate / self.chunk * self.seconds_per_window)

    def start(self) -> None:
        if self.stream is not None:
            return

        logger.info("Starting microphone stream")
        self.stream = self.audio.open(
            input=True,
            channels=self.channels,
            format=self.format_type,
            rate=self.rate,
            frames_per_buffer=self.chunk,
        )

    def read_chunk(self) -> bytes:
        if self.stream is None:
            raise RuntimeError("Microphone stream has not been started.")

        return self.stream.read(num_frames=self.chunk, exception_on_overflow=False)

    def read_window(self) -> bytes | None:
        """
        Read one chunk from the microphone.
        Return a full buffered window of audio bytes when ready.
        Otherwise return None.
        """
        data = self.read_chunk()
        self.frames.append(data)

        if len(self.frames) >= self.chunks_per_window:
            window = b"".join(self.frames)
            self.frames = []
            return window

        return None

    def stop(self) -> None:
        if self.stream is not None:
            logger.info("Stopping microphone stream")
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def terminate(self) -> None:
        self.stop()
        logger.info("Terminating microphone backend")
        self.audio.terminate()
