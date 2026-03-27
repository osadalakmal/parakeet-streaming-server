from __future__ import annotations

import sys
import types
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub out mlx_whisper so tests run without real model weights.
# ---------------------------------------------------------------------------
_fake_mlx_whisper = types.ModuleType("mlx_whisper")
_call_count = 0
_last_audio_size = 0


def _fake_transcribe(audio, *, path_or_hf_repo="", **kwargs):
    global _call_count, _last_audio_size
    _call_count += 1
    _last_audio_size = audio.size
    return {"text": f"result-{_call_count}"}


_fake_mlx_whisper.transcribe = _fake_transcribe
sys.modules.setdefault("mlx_whisper", _fake_mlx_whisper)

from app.session import StreamConfig, StreamingSession  # noqa: E402


@pytest.fixture(autouse=True)
def reset_counters():
    global _call_count, _last_audio_size
    _call_count = 0
    _last_audio_size = 0
    yield


def _pcm(values: list[int]) -> bytes:
    return np.array(values, dtype=np.int16).tobytes()


def test_add_pcm16_chunk_returns_partial_without_transcribing() -> None:
    session = StreamingSession("fake/model", StreamConfig())
    session.start()

    payload = session.add_pcm16_chunk(_pcm([0, 100, 200]))

    assert payload["type"] == "partial"
    assert payload["is_final"] is False
    # No transcription should have happened yet.
    assert _call_count == 0


def test_flush_triggers_transcription_and_returns_partial() -> None:
    session = StreamingSession("fake/model", StreamConfig())
    session.start()
    session.add_pcm16_chunk(_pcm([1, 2]))
    session.add_pcm16_chunk(_pcm([3, 4]))

    payload = session.flush()

    assert payload["type"] == "partial"
    assert payload["is_final"] is False
    assert payload["text"] == "result-1"
    assert _call_count == 1
    # All buffered samples were passed to transcribe.
    assert _last_audio_size == 4


def test_stop_triggers_transcription_and_returns_final() -> None:
    session = StreamingSession("fake/model", StreamConfig())
    session.start()
    session.add_pcm16_chunk(_pcm([10, 20]))

    payload = session.stop()

    assert payload == {"type": "final", "text": "result-1", "is_final": True}
    assert _call_count == 1


def test_stop_with_empty_buffer_returns_empty_final() -> None:
    session = StreamingSession("fake/model", StreamConfig())
    session.start()

    payload = session.stop()

    assert payload["type"] == "final"
    assert payload["text"] == ""
    assert _call_count == 0


def test_language_and_beam_size_forwarded_to_transcribe(monkeypatch) -> None:
    captured: dict = {}

    def recording_transcribe(audio, *, path_or_hf_repo="", **kwargs):
        captured.update(kwargs)
        return {"text": "ok"}

    monkeypatch.setattr(_fake_mlx_whisper, "transcribe", recording_transcribe)

    config = StreamConfig(language="en", beam_size=8)
    session = StreamingSession("fake/model", config)
    session.start()
    session.add_pcm16_chunk(_pcm([1, 2]))
    session.flush()

    assert captured.get("language") == "en"
    assert captured.get("beam_size") == 8
