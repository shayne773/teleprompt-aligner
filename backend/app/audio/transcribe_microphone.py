"""Simple command-line runner for buffered-streaming microphone transcription."""

from __future__ import annotations

import math
import logging
from collections import deque

from backend.app.audio.microphone import MicrophoneCapture
from backend.app.audio.nemo_asr import NemoASRTranscriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

ROLLING_WINDOW_SECONDS = 1.0
INFERENCE_STRIDE_SECONDS = 0.25
BYTES_PER_SAMPLE = 2
CHANNELS = 1


def merge_transcript(previous: str, current: str) -> tuple[str, str]:
    previous = previous.strip()
    current = current.strip()

    if not current:
        return previous, ""
    if current == previous:
        return previous, ""
    if previous and current.startswith(previous):
        return current, current[len(previous) :]
    return current, current


def main() -> None:
    """Capture live microphone audio and print buffered transcript updates."""
    mic = MicrophoneCapture(
        channels=CHANNELS,
        rate=16000,
        chunk=256,
        seconds_per_window=3,
    )

    try:
        asr = NemoASRTranscriber()
    except Exception:
        logger.exception("Unable to initialize ASR transcriber; exiting.")
        return

    mic.start()
    logger.info(
        "Buffered streaming started (rolling_window=%.2fs, stride=%.2fs).",
        ROLLING_WINDOW_SECONDS,
        INFERENCE_STRIDE_SECONDS,
    )
    logger.info("Listening... Press Ctrl+C to stop.")

    max_buffer_bytes = int(
        mic.rate * ROLLING_WINDOW_SECONDS * BYTES_PER_SAMPLE * mic.channels
    )
    chunks_per_inference = max(
        1, math.ceil(INFERENCE_STRIDE_SECONDS / mic.chunk_duration_seconds)
    )
    rolling_chunks: deque[bytes] = deque()
    rolling_bytes = 0
    chunks_since_last_inference = 0
    last_transcript = ""

    try:
        while True:
            chunk = mic.read_chunk()
            rolling_chunks.append(chunk)
            rolling_bytes += len(chunk)
            chunks_since_last_inference += 1

            while rolling_bytes > max_buffer_bytes and rolling_chunks:
                removed = rolling_chunks.popleft()
                rolling_bytes -= len(removed)

            if chunks_since_last_inference < chunks_per_inference:
                continue

            try:
                window = b"".join(rolling_chunks)
                text = asr.transcribe_window(window, sample_rate=mic.rate)
            except Exception:
                logger.exception("Unexpected transcription loop failure")
                continue
            finally:
                chunks_since_last_inference = 0

            last_transcript, delta = merge_transcript(last_transcript, text)
            if not delta:
                continue

            if delta == last_transcript:
                print(f"Transcript: {delta}")
            else:
                print(f"Transcript update: {delta}")
    except KeyboardInterrupt:
        logger.info("Stopping microphone transcription.")
    finally:
        mic.terminate()
        logger.info("Microphone resources released.")


if __name__ == "__main__":
    main()
