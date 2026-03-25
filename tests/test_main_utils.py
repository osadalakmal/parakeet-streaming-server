from __future__ import annotations

import importlib
import sys
import types


def load_main_module():
    fake_parakeet = types.ModuleType("parakeet_mlx")
    fake_parakeet.from_pretrained = lambda _name: object()
    sys.modules.setdefault("parakeet_mlx", fake_parakeet)
    return importlib.import_module("app.main")


def test_parse_json_accepts_dict_payload() -> None:
    main = load_main_module()
    assert main._parse_json('{"type":"start"}') == {"type": "start"}


def test_parse_json_rejects_non_dict_or_invalid_json() -> None:
    main = load_main_module()
    assert main._parse_json("[]") is None
    assert main._parse_json("not-json") is None
