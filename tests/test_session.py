from __future__ import annotations

import numpy as np

from app.session import StreamConfig, StreamingSession


class FakeResult:
    def __init__(self) -> None:
        self.text = ""
        self.finalized_text = ""


class FakeTranscriber:
    def __init__(self) -> None:
        self.result = FakeResult()
        self._chunks = 0

    def add_audio(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            self.result.finalized_text = self.result.text
            return

        self._chunks += 1
        self.result.text = f"chunk-{self._chunks}"
        if self._chunks > 1:
            self.result.finalized_text = f"chunk-{self._chunks - 1}"


class FakeContextManager:
    def __init__(self, transcriber: FakeTranscriber) -> None:
        self.transcriber = transcriber

    def __enter__(self) -> FakeTranscriber:
        return self.transcriber

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class FakeModel:
    def __init__(self) -> None:
        self.kwargs = None
        self.transcriber = FakeTranscriber()

    def transcribe_stream(self, **kwargs):
        self.kwargs = kwargs
        return FakeContextManager(self.transcriber)


def test_streaming_session_start_and_partial_updates() -> None:
    model = FakeModel()
    session = StreamingSession(
        model,
        StreamConfig(sample_rate=16_000, context_size=128, depth=4, keep_original_attention=True),
    )

    session.start()
    first = session.add_pcm16_chunk(np.array([0, 10], dtype=np.int16).tobytes())
    second = session.add_pcm16_chunk(np.array([10, 20], dtype=np.int16).tobytes())

    assert model.kwargs == {
        "sample_rate": 16_000,
        "context_size": 128,
        "depth": 4,
        "keep_original_attention": True,
    }
    assert first["type"] == "partial"
    assert first["text"] == "chunk-1"
    assert second["finalized_text"] == "chunk-1"


def test_streaming_session_stop_returns_final_payload() -> None:
    model = FakeModel()
    session = StreamingSession(model, StreamConfig())

    session.start()
    session.add_pcm16_chunk(np.array([1, 2], dtype=np.int16).tobytes())

    final_payload = session.stop()

    assert final_payload == {"type": "final", "text": "chunk-1", "is_final": True}
