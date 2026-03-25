from __future__ import annotations

import numpy as np


def pcm16le_to_float32_mono(data: bytes) -> np.ndarray:
    """Decode raw PCM16LE mono bytes into float32 samples in [-1.0, 1.0]."""
    if len(data) % 2 != 0:
        raise ValueError("PCM16 payload must contain an even number of bytes.")

    int16_samples = np.frombuffer(data, dtype="<i2")
    if int16_samples.size == 0:
        return np.empty(0, dtype=np.float32)

    return (int16_samples.astype(np.float32) / 32768.0).copy()
