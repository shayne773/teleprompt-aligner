"""Simple command-line runner for NeMo cache-aware streaming microphone transcription."""

from __future__ import annotations

import logging

from backend.app.audio.microphone import MicrophoneCapture
from backend.app.audio.nemo_asr import NemoASRTranscriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK_SIZE = 256


def main() -> None:
    """Capture live microphone audio and print streaming transcript updates."""
    mic = MicrophoneCapture(
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        chunk=CHUNK_SIZE,
    )

    try:
        asr = NemoASRTranscriber()
        asr.start_stream()
    except Exception:
        logger.exception("Unable to initialize NeMo streaming transcriber; exiting.")
        return

    mic.start()
    logger.info("Streaming transcription started.")
    logger.info("Listening... Press Ctrl+C to stop.")

    last_printed_text = ""

    try:
        while True:
            chunk = mic.read_chunk()
            text = asr.process_chunk(chunk, sample_rate=mic.rate)
            if not text or text == last_printed_text:
                continue

            print(f"Transcript: {text}")
            last_printed_text = text
    except KeyboardInterrupt:
        logger.info("Stopping microphone transcription.")
    finally:
        try:
            final_text = asr.finish_stream()
            if final_text and final_text != last_printed_text:
                print(f"Final transcript: {final_text}")
        except Exception:
            logger.exception("Failed to finalize streaming session cleanly.")
        finally:
            mic.terminate()
            logger.info("Microphone resources released.")


if __name__ == "__main__":
    main()
