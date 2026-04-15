"""NVIDIA NeMo cache-aware streaming speech-to-text helpers."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Optional

import nemo.collections.asr as nemo_asr
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Cache-aware streaming FastConformer RNNT model.
DEFAULT_STREAMING_MODEL = "stt_en_fastconformer_transducer_large_streaming"


class NemoASRTranscriber:
    """Stateful NeMo cache-aware streaming transcriber for PCM16 chunks."""

    def __init__(
        self,
        model_name: str = DEFAULT_STREAMING_MODEL,
        device: Optional[str] = None,
        silence_threshold: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.silence_threshold = silence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            "Loading NeMo streaming ASR model '%s' on device '%s'...",
            self.model_name,
            self.device,
        )
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
        model.eval()
        self.model = model.to(self.device)

        if not hasattr(self.model, "conformer_stream_step"):
            raise RuntimeError(
                "Loaded model does not expose `conformer_stream_step`; "
                "please use a cache-aware streaming NeMo model."
            )

        self._stream_active = False
        self._cache_last_channel: torch.Tensor | None = None
        self._cache_last_time: torch.Tensor | None = None
        self._cache_last_channel_len: torch.Tensor | None = None
        self._last_text = ""

        logger.info("NeMo streaming model loaded successfully.")

    @staticmethod
    def pcm16_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        if samples.size == 0:
            return np.array([], dtype=np.float32)
        return np.ravel(samples.astype(np.float32) / 32768.0)

    @staticmethod
    def rms(signal: np.ndarray) -> float:
        if signal.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(signal), dtype=np.float32)))

    @staticmethod
    def _extract_text(result: Any) -> str:
        if result is None:
            return ""

        if isinstance(result, str):
            return result.strip()

        if isinstance(result, (list, tuple)):
            for item in result:
                text = NemoASRTranscriber._extract_text(item)
                if text:
                    return text
            return ""

        text = getattr(result, "text", None)
        if isinstance(text, str):
            return text.strip()

        return ""

    def _init_stream_cache(self) -> None:
        if not hasattr(self.model, "encoder") or not hasattr(
            self.model.encoder, "get_initial_cache_state"
        ):
            raise RuntimeError(
                "Model encoder does not expose `get_initial_cache_state`; "
                "cache-aware streaming is unavailable for this model/version."
            )

        cache = self.model.encoder.get_initial_cache_state(
            batch_size=1,
            device=self.device,
        )
        (
            self._cache_last_channel,
            self._cache_last_time,
            self._cache_last_channel_len,
        ) = cache

    def start_stream(self) -> None:
        self._init_stream_cache()
        self._last_text = ""
        self._stream_active = True
        logger.info("NeMo streaming ASR session started.")

    def _stream_step(self, signal: torch.Tensor, signal_len: torch.Tensor, *, is_last: bool) -> Any:
        params = inspect.signature(self.model.conformer_stream_step).parameters
        kwargs: dict[str, Any] = {}

        if "processed_signal" in params:
            kwargs["processed_signal"] = signal
        elif "audio_signal" in params:
            kwargs["audio_signal"] = signal

        if "processed_signal_length" in params:
            kwargs["processed_signal_length"] = signal_len
        elif "audio_signal_length" in params:
            kwargs["audio_signal_length"] = signal_len

        if "cache_last_channel" in params:
            kwargs["cache_last_channel"] = self._cache_last_channel
        if "cache_last_time" in params:
            kwargs["cache_last_time"] = self._cache_last_time
        if "cache_last_channel_len" in params:
            kwargs["cache_last_channel_len"] = self._cache_last_channel_len

        if "keep_all_outputs" in params:
            kwargs["keep_all_outputs"] = False
        if "return_transcription" in params:
            kwargs["return_transcription"] = True
        if "is_last" in params:
            kwargs["is_last"] = is_last
        if "drop_extra_pre_encoded" in params:
            kwargs["drop_extra_pre_encoded"] = not is_last

        return self.model.conformer_stream_step(**kwargs)

    def _handle_stream_result(self, result: Any) -> str:
        if isinstance(result, tuple) and len(result) >= 4:
            maybe_text = result[0]
            self._cache_last_channel = result[1]
            self._cache_last_time = result[2]
            self._cache_last_channel_len = result[3]
            text = self._extract_text(maybe_text)
        else:
            text = self._extract_text(result)

        if not text:
            return ""

        self._last_text = text
        return text

    def process_chunk(self, audio_bytes: bytes, sample_rate: int) -> str:
        if not self._stream_active:
            raise RuntimeError("Streaming session is not active. Call start_stream() first.")

        if sample_rate != 16000:
            raise ValueError(f"Expected 16000Hz PCM16 audio, got {sample_rate}Hz")

        waveform = self.pcm16_bytes_to_float32(audio_bytes)
        if waveform.size == 0:
            return ""

        if self.silence_threshold > 0 and self.rms(waveform) < self.silence_threshold:
            return ""

        signal = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
        signal_len = torch.tensor([waveform.shape[0]], dtype=torch.long, device=self.device)

        with torch.no_grad():
            result = self._stream_step(signal, signal_len, is_last=False)
        return self._handle_stream_result(result)

    def finish_stream(self) -> str:
        if not self._stream_active:
            return ""

        empty_signal = torch.zeros((1, 0), dtype=torch.float32, device=self.device)
        empty_len = torch.tensor([0], dtype=torch.long, device=self.device)

        with torch.no_grad():
            result = self._stream_step(empty_signal, empty_len, is_last=True)

        final_text = self._handle_stream_result(result)
        self._stream_active = False
        logger.info("NeMo streaming ASR session finalized.")
        return final_text

    def reset_stream(self) -> None:
        self._stream_active = False
        self._cache_last_channel = None
        self._cache_last_time = None
        self._cache_last_channel_len = None
        self._last_text = ""
