"""NVIDIA NeMo cache-aware streaming speech-to-text helpers."""

from __future__ import annotations
from omegaconf import OmegaConf
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

import copy
import inspect
import logging
from typing import Any, Optional

import nemo.collections.asr as nemo_asr
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Cache-aware streaming FastConformer RNNT model.
DEFAULT_STREAMING_MODEL = "stt_en_fastconformer_hybrid_large_streaming_multi"


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
        
        cfg = copy.deepcopy(self.model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)

        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"

        self._preprocessor = EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
        self._preprocessor.to(self.device)

        self._stream_active = False
        self._cache_last_channel: torch.Tensor | None = None
        self._cache_last_time: torch.Tensor | None = None
        self._cache_last_channel_len: torch.Tensor | None = None
        self._last_text = ""

        self._previous_hypotheses: Any = None
        self._pred_out_stream: torch.Tensor | None = None
        self._cache_pre_encode: torch.Tensor | None = None
        self._feature_dim = int(self.model.cfg.preprocessor.features)
        self._pre_encode_cache_size = self.model.encoder.streaming_cfg.pre_encode_cache_size[1]
        self._cache_pre_encode = torch.zeros(
            (1, self._feature_dim, self._pre_encode_cache_size),
            dtype=torch.float32,
            device=self.device,
        )

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
    
    def preprocess_audio(self, audio):
        device = self.device
        audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(device)
        audio_signal_len = torch.Tensor([audio.shape[0]]).to(device)
        processed_signal, processed_signal_length = self._preprocessor(input_signal=audio_signal, length=audio_signal_len)
        return processed_signal, processed_signal_length

    def start_stream(self) -> None:
        self._init_stream_cache()
        self._previous_hypotheses = None
        self._pred_out_stream = None
        self._cache_pre_encode = torch.zeros(
            (1, self._feature_dim, self._pre_encode_cache_size),
            dtype=torch.float32,
            device=self.device,
        )
        self._last_text = ""
        self._stream_active = True
        logger.info("NeMo streaming ASR session started.")

    def _stream_step(self, processed_signal: torch.Tensor, processed_signal_length: torch.Tensor) -> Any:

        return self.model.conformer_stream_step(
            processed_signal = processed_signal,
            processed_signal_length = processed_signal_length,
            cache_last_channel = self._cache_last_channel,
            cache_last_time = self._cache_last_time,
            cache_last_channel_len = self._cache_last_channel_len,
            keep_all_outputs = False,
            previous_hypotheses = self._previous_hypotheses,
            previous_pred_out = self._pred_out_stream,
            drop_extra_pre_encoded = None,
            return_transcription = True
        )

    def _handle_stream_result(self, hyps) -> list[str]:
        if hyps is None:
            return []

        if isinstance(hyps, Hypothesis):
            return [hyps.text]

        if isinstance(hyps, (list, tuple)):
            if len(hyps) == 0:
                return []
            if isinstance(hyps[0], Hypothesis):
                return [hyp.text for hyp in hyps]
            return [str(x).strip() for x in hyps if str(x).strip()]

        text = getattr(hyps, "text", None)
        if isinstance(text, str) and text.strip():
            return [text.strip()]

        return []


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

        processed_signal, processed_signal_length = self.preprocess_audio(audio=waveform)
        processed_signal = torch.cat([self._cache_pre_encode, processed_signal], dim=-1)
        processed_signal_length += self._cache_pre_encode.shape[1]

        self._cache_pre_encode = processed_signal[:, :, -self._pre_encode_cache_size:]

        with torch.no_grad():
            (
                self._pred_out_stream,
                transcribed_texts,
                self._cache_last_channel,
                self._cache_last_time,
                self._cache_last_channel_len,
                self._previous_hypotheses
            ) = self._stream_step(processed_signal, processed_signal_length)
        
        texts  = self._handle_stream_result(transcribed_texts)
        text = texts[0] if texts else ""
        self._last_text = text

        return text

    def finish_stream(self) -> str:
        if not self._stream_active:
            return ""

        self._stream_active = False
        logger.info("NeMo streaming ASR session finalized.")
        return self._last_text or ""

    def reset_stream(self) -> None:
        self._stream_active = False
        self._cache_last_channel = None
        self._cache_last_time = None
        self._cache_last_channel_len = None
        self._previous_hypotheses = None
        self._pred_out_stream = None
        self._cache_pre_encode = None
        self._last_text = ""
