from __future__ import annotations

import importlib
import sys
import types


def load_main_module(monkeypatch):
    fake_mlx_whisper = types.ModuleType("mlx_whisper")
    # transcribe is called at startup (warm-up) and during sessions; patch it
    # to return an empty result so tests don't require real model weights.
    fake_mlx_whisper.transcribe = lambda *_args, **_kwargs: {"text": ""}
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mlx_whisper)
    monkeypatch.delitem(sys.modules, "app.main", raising=False)
    return importlib.import_module("app.main")


def test_parse_json_accepts_dict_json(monkeypatch) -> None:
    main = load_main_module(monkeypatch)
    assert main._parse_json('{"type":"start"}') == {"type": "start"}


def test_parse_json_rejects_non_dict_or_invalid_json(monkeypatch) -> None:
    main = load_main_module(monkeypatch)
    assert main._parse_json("[]") is None
    assert main._parse_json("not-json") is None
