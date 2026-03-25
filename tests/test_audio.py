from __future__ import annotations

import numpy as np
import pytest

from app.audio import pcm16le_to_float32_mono


def test_pcm16le_to_float32_mono_decodes_expected_values() -> None:
    pcm = np.array([-32768, 0, 32767], dtype="<i2").tobytes()

    decoded = pcm16le_to_float32_mono(pcm)

    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, np.array([-1.0, 0.0, 32767 / 32768], dtype=np.float32))


def test_pcm16le_to_float32_mono_rejects_odd_byte_count() -> None:
    with pytest.raises(ValueError, match="even number of bytes"):
        pcm16le_to_float32_mono(b"\x01")
