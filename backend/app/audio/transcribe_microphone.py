"""Simple command-line runner for offline-per-window microphone transcription."""

from __future__ import annotations

import logging

from backend.app.audio.microphone import MicrophoneCapture
from backend.app.audio.nemo_asr import NemoASRTranscriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Capture live microphone audio and print transcript lines per full window."""
    mic = MicrophoneCapture(
        channels=1,
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
    logger.info("Listening... Press Ctrl+C to stop.")

    try:
        while True:
            window = mic.read_window()
            if window is None:
                continue

            try:
                text = asr.transcribe_window(window, sample_rate=mic.rate)
            except Exception:
                logger.exception("Unexpected transcription loop failure")
                continue

            if text:
                print(f"Transcript: {text}")
    except KeyboardInterrupt:
        logger.info("Stopping microphone transcription.")
    finally:
        mic.terminate()
        logger.info("Microphone resources released.")


if __name__ == "__main__":
    main()
