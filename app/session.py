from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.audio import pcm16le_to_float32_mono


@dataclass(slots=True)
class StreamConfig:
    sample_rate: int = 16_000
    context_size: int = 256
    depth: int = 8
    keep_original_attention: bool = False


class StreamingSession:
    """Owns one streaming transcriber context for a single websocket session."""

    def __init__(self, model: Any, config: StreamConfig):
        self._model = model
        self.config = config
        self._ctx = None
        self._transcriber = None
        self.finalized_text = ""

    def start(self) -> None:
        if self._transcriber is not None:
            raise RuntimeError("Session already started.")

        self._ctx = self._model.transcribe_stream(
            sample_rate=self.config.sample_rate,
            context_size=(self.config.context_size, self.config.context_size),
            depth=self.config.depth,
            keep_original_attention=self.config.keep_original_attention,
        )
        self._transcriber = self._ctx.__enter__()

    def add_pcm16_chunk(self, chunk: bytes) -> dict[str, str | bool]:
        if self._transcriber is None:
            raise RuntimeError("Session has not been started.")

        audio = pcm16le_to_float32_mono(chunk)
        if audio.size > 0:
            self._transcriber.add_audio(audio)

        return self._current_partial_payload()

    def flush(self) -> dict[str, str | bool]:
        if self._transcriber is None:
            raise RuntimeError("Session has not been started.")

        # Some streaming backends flush on empty audio chunks.
        self._transcriber.add_audio(np.empty(0, dtype=np.float32))
        return self._current_partial_payload()

    def stop(self) -> dict[str, str | bool]:
        if self._transcriber is None:
            return {"type": "final", "text": self.finalized_text, "is_final": True}

        try:
            self._transcriber.add_audio(np.empty(0, dtype=np.float32))
            result = self._safe_result()
            final_text = self._extract_text(result)
            if final_text:
                self.finalized_text = final_text
            return {"type": "final", "text": self.finalized_text, "is_final": True}
        finally:
            self.close()

    def close(self) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
        self._ctx = None
        self._transcriber = None

    def _current_partial_payload(self) -> dict[str, str | bool]:
        result = self._safe_result()
        text = self._extract_text(result)
        finalized = self._extract_finalized_text(result)
        if finalized:
            self.finalized_text = finalized
        elif text and text != self.finalized_text:
            # Fallback if API does not expose a separate finalized text field.
            self.finalized_text = text

        return {
            "type": "partial",
            "text": text,
            "finalized_text": self.finalized_text,
            "is_final": False,
        }

    def _safe_result(self) -> Any:
        return getattr(self._transcriber, "result", None)

    @staticmethod
    def _extract_text(result: Any) -> str:
        if result is None:
            return ""

        text = getattr(result, "text", "")
        return text if isinstance(text, str) else ""

    @staticmethod
    def _extract_finalized_text(result: Any) -> str:
        if result is None:
            return ""

        for candidate in ("finalized_text", "final_text", "stable_text"):
            value = getattr(result, candidate, "")
            if isinstance(value, str) and value:
                return value
        return ""
