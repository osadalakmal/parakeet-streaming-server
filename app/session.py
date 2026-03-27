from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx_whisper
import numpy as np

from app.audio import pcm16le_to_float32_mono


@dataclass(slots=True)
class StreamConfig:
    sample_rate: int = 16_000
    language: str | None = None  # None = auto-detect
    beam_size: int = 5


class StreamingSession:
    """Owns one Whisper transcription session for a single websocket connection.

    Audio is buffered in memory.  Transcription runs on explicit flush/stop
    so that the (blocking) Whisper call never happens on the hot audio path.
    Callers are responsible for running flush() and stop() in a thread to
    avoid blocking the asyncio event loop.
    """

    def __init__(self, model_path: str, config: StreamConfig) -> None:
        self._model_path = model_path
        self.config = config
        self._chunks: list[np.ndarray] = []
        self._last_text: str = ""
        self.finalized_text: str = ""

    def start(self) -> None:
        # Nothing to set up; audio buffering starts on first add_pcm16_chunk call.
        pass

    def add_pcm16_chunk(self, chunk: bytes) -> dict[str, Any]:
        """Buffer the PCM chunk and return the last known partial transcript."""
        audio = pcm16le_to_float32_mono(chunk)
        if audio.size > 0:
            self._chunks.append(audio)
        return self._partial_payload()

    def flush(self) -> dict[str, Any]:
        """Transcribe the current buffer and return a partial result.

        Blocking — run in a thread from the async layer.
        """
        self._run_transcription()
        return self._partial_payload()

    def stop(self) -> dict[str, Any]:
        """Transcribe the current buffer and return the final result.

        Blocking — run in a thread from the async layer.
        """
        self._run_transcription()
        self.finalized_text = self._last_text
        return {"type": "final", "text": self.finalized_text, "is_final": True}

    def close(self) -> None:
        self._chunks.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_audio(self) -> np.ndarray:
        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self._chunks)

    def _run_transcription(self) -> None:
        audio = self._get_audio()
        if audio.size == 0:
            return

        decode_options: dict[str, Any] = {"beam_size": self.config.beam_size}
        if self.config.language is not None:
            decode_options["language"] = self.config.language

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model_path,
            **decode_options,
        )
        self._last_text = (result.get("text") or "").strip()

    def _partial_payload(self) -> dict[str, Any]:
        return {
            "type": "partial",
            "text": self._last_text,
            "finalized_text": self.finalized_text,
            "is_final": False,
        }
