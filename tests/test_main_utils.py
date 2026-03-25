from __future__ import annotations

import importlib
import sys
import types


def load_main_module(monkeypatch):
    fake_parakeet = types.ModuleType("parakeet_mlx")
    fake_parakeet.from_pretrained = lambda _name: object()
    # Use monkeypatch so changes to sys.modules are reverted after each test
    monkeypatch.setitem(sys.modules, "parakeet_mlx", fake_parakeet)
    monkeypatch.delitem(sys.modules, "app.main", raising=False)
    return importlib.import_module("app.main")


def test_parse_json_accepts_dict_payload(monkeypatch) -> None:
    main = load_main_module(monkeypatch)
    assert main._parse_json('{"type":"start"}') == {"type": "start"}


def test_parse_json_rejects_non_dict_or_invalid_json(monkeypatch) -> None:
    main = load_main_module(monkeypatch)
    assert main._parse_json("[]") is None
    assert main._parse_json("not-json") is None
