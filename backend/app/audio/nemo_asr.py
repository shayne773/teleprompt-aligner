"""NVIDIA NeMo offline-per-window speech-to-text helpers."""

from __future__ import annotations

import logging
import tempfile
import wave
from pathlib import Path
from typing import Any, Optional

import nemo.collections.asr as nemo_asr
import numpy as np
import torch

logger = logging.getLogger(__name__)


class NemoASRTranscriber:
    """Transcribe PCM16 microphone windows with a pretrained NVIDIA NeMo ASR model."""

    def __init__(
        self,
        model_name: str = "stt_en_fastconformer_transducer_large",
        device: Optional[str] = None,
        silence_threshold: float = 0.01,
    ) -> None:
        self.model_name = model_name
        self.silence_threshold = silence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            "Loading NeMo ASR model '%s' on device '%s'...", self.model_name, self.device
        )
        try:
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
            model.eval()
            self.model = model.to(self.device)
        except Exception:
            logger.exception("Failed to load NeMo ASR model '%s'", self.model_name)
            raise

        logger.info("NeMo ASR model loaded successfully.")

    @staticmethod
    def pcm16_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
        """Convert little-endian PCM16 mono bytes to a normalized float32 waveform."""
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        if samples.size == 0:
            return np.array([], dtype=np.float32)

        waveform = samples.astype(np.float32) / 32768.0
        return np.ravel(waveform)

    @staticmethod
    def rms(signal: np.ndarray) -> float:
        """Return root-mean-square energy for a waveform."""
        if signal.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(signal), dtype=np.float32)))

    @staticmethod
    def extract_text(result: Any) -> str:
        """Best-effort extraction of transcript text from NeMo outputs."""
        if result is None:
            return ""

        if isinstance(result, str):
            return result.strip()

        text = getattr(result, "text", None)
        if isinstance(text, str):
            return text.strip()

        return str(result).strip()

    def _normalize_transcribe_output(self, results: Any) -> str:
        if isinstance(results, (list, tuple)):
            if not results:
                return ""
            return self.extract_text(results[0])

        return self.extract_text(results)

    def _transcribe_via_temp_wav(self, audio_bytes: bytes, sample_rate: int) -> str:
        """Fallback transcription path for NeMo models expecting filepath inputs."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with wave.open(str(tmp_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_bytes)

            results = self.model.transcribe([str(tmp_path)], batch_size=1)
            return self._normalize_transcribe_output(results)
        except Exception:
            logger.exception("NeMo WAV-file fallback transcription failed")
            return ""
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed to delete temporary WAV file: %s", tmp_path)

    def transcribe_window(self, audio_bytes: bytes, sample_rate: int) -> str:
        """Transcribe a single completed microphone window into text."""
        waveform = self.pcm16_bytes_to_float32(audio_bytes)
        if waveform.size == 0:
            return ""

        if self.rms(waveform) < self.silence_threshold:
            return ""

        try:
            results = self.model.transcribe([waveform], batch_size=1)
            text = self._normalize_transcribe_output(results)
            if text:
                return text

            logger.debug("Direct waveform transcription returned no text; trying WAV fallback")
            return self._transcribe_via_temp_wav(audio_bytes, sample_rate)
        except Exception:
            logger.exception("NeMo direct waveform transcription failed; trying WAV fallback")
            return self._transcribe_via_temp_wav(audio_bytes, sample_rate)
