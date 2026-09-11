"""
tests/unit/test_pcm.py
========================
audio_processing.pcm: chunk framing + PCM16<->float32 encode/decode.
Pure functions, no audio hardware, no network, no ASR.
"""

import numpy as np
import pytest

from audio_processing.pcm import (
    chunk_samples,
    float32_to_pcm16_bytes,
    frame_waveform,
    pcm16_bytes_to_float32,
)


# ---------------------------------------------------------------------------
# chunk_samples
# ---------------------------------------------------------------------------

def test_chunk_samples_within_20_to_40ms_range_at_16k():
    # Requirement: chunk duration configurable, starting in the 20-40ms range.
    assert chunk_samples(16000, chunk_ms=20) == 320
    assert chunk_samples(16000, chunk_ms=30) == 480
    assert chunk_samples(16000, chunk_ms=40) == 640


def test_chunk_samples_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        chunk_samples(0, 30)
    with pytest.raises(ValueError):
        chunk_samples(16000, 0)
    with pytest.raises(ValueError):
        chunk_samples(16000, -5)


# ---------------------------------------------------------------------------
# frame_waveform (chunk framing)
# ---------------------------------------------------------------------------

def test_frame_waveform_splits_into_expected_chunk_count():
    sr = 16000
    size = chunk_samples(sr, 30)
    waveform = np.arange(size * 5, dtype=np.float32)
    chunks = frame_waveform(waveform, sr, chunk_ms=30)
    assert len(chunks) == 5
    for c in chunks:
        assert len(c) == size


def test_frame_waveform_last_chunk_can_be_short_not_padded():
    sr = 16000
    size = chunk_samples(sr, 30)
    waveform = np.arange(size * 3 + 17, dtype=np.float32)  # not an exact multiple
    chunks = frame_waveform(waveform, sr, chunk_ms=30)
    assert len(chunks) == 4
    assert len(chunks[0]) == size
    assert len(chunks[1]) == size
    assert len(chunks[2]) == size
    assert len(chunks[3]) == 17  # short, not zero-padded to `size`


def test_frame_waveform_preserves_sample_order():
    sr = 16000
    waveform = np.arange(1000, dtype=np.float32)
    chunks = frame_waveform(waveform, sr, chunk_ms=20)
    reassembled = np.concatenate(chunks)
    np.testing.assert_array_equal(reassembled, waveform)


def test_frame_waveform_empty_input_yields_no_chunks():
    assert frame_waveform(np.zeros(0, dtype=np.float32), 16000, chunk_ms=30) == []


# ---------------------------------------------------------------------------
# PCM16 <-> float32 round trip
# ---------------------------------------------------------------------------

def test_pcm16_round_trip_preserves_signal_within_quantization_error():
    original = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 0.25], dtype=np.float32)
    encoded = float32_to_pcm16_bytes(original)
    assert len(encoded) == len(original) * 2  # 2 bytes per int16 sample
    decoded = pcm16_bytes_to_float32(encoded)
    np.testing.assert_allclose(decoded, original, atol=1e-3)


def test_pcm16_clips_out_of_range_values():
    original = np.array([2.0, -2.0], dtype=np.float32)
    encoded = float32_to_pcm16_bytes(original)
    decoded = pcm16_bytes_to_float32(encoded)
    np.testing.assert_allclose(decoded, [1.0, -1.0], atol=1e-3)


def test_pcm16_decode_rejects_odd_byte_count():
    """Malformed/incomplete audio: an odd number of bytes cannot be whole
    16-bit samples and must be rejected, not silently misinterpreted."""
    malformed = b"\x00\x01\x02"  # 3 bytes
    with pytest.raises(ValueError):
        pcm16_bytes_to_float32(malformed)


def test_pcm16_decode_empty_bytes_yields_empty_array():
    decoded = pcm16_bytes_to_float32(b"")
    assert decoded.shape == (0,)
