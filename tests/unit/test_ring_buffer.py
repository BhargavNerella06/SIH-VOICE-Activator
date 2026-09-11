"""
tests/unit/test_ring_buffer.py
================================
audio_processing.ring_buffer.AudioRingBuffer: the generic circular
pre-roll buffer abstraction. Deliberately tested standalone, with no
keyword_spotting import anywhere -- see the module's own docstring for
why it must stay decoupled from any specific detector.
"""

import numpy as np
import pytest

from audio_processing.ring_buffer import AudioRingBuffer


def test_rejects_non_positive_capacity_or_sample_rate():
    with pytest.raises(ValueError):
        AudioRingBuffer(capacity_seconds=0, sample_rate=16000)
    with pytest.raises(ValueError):
        AudioRingBuffer(capacity_seconds=-1, sample_rate=16000)
    with pytest.raises(ValueError):
        AudioRingBuffer(capacity_seconds=1.0, sample_rate=0)


def test_capacity_samples_computed_from_seconds_and_rate():
    buf = AudioRingBuffer(capacity_seconds=0.5, sample_rate=16000)
    assert buf.capacity_samples == 8000


def test_empty_buffer_reads_as_empty():
    buf = AudioRingBuffer(capacity_seconds=0.5, sample_rate=16000)
    assert buf.filled_samples == 0
    assert buf.duration_seconds == 0.0
    assert buf.is_full is False
    np.testing.assert_array_equal(buf.read(), np.zeros(0, dtype=np.float32))


def test_partial_fill_returns_only_written_samples_in_order():
    buf = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)  # capacity = 10 samples
    buf.write(np.array([1, 2, 3], dtype=np.float32))
    assert buf.filled_samples == 3
    assert buf.is_full is False
    np.testing.assert_array_equal(buf.read(), [1, 2, 3])


def test_multiple_writes_accumulate_in_order_before_full():
    buf = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)  # capacity = 10
    buf.write(np.array([1, 2], dtype=np.float32))
    buf.write(np.array([3, 4, 5], dtype=np.float32))
    np.testing.assert_array_equal(buf.read(), [1, 2, 3, 4, 5])
    assert buf.filled_samples == 5


def test_oldest_samples_are_overwritten_once_full():
    buf = AudioRingBuffer(capacity_seconds=1.0, sample_rate=5)  # capacity = 5 samples
    buf.write(np.array([1, 2, 3, 4, 5], dtype=np.float32))
    assert buf.is_full is True
    buf.write(np.array([6, 7], dtype=np.float32))
    # Oldest (1, 2) dropped; buffer now holds the most recent 5 samples in order.
    np.testing.assert_array_equal(buf.read(), [3, 4, 5, 6, 7])
    assert buf.filled_samples == 5


def test_write_larger_than_capacity_keeps_only_the_tail():
    buf = AudioRingBuffer(capacity_seconds=1.0, sample_rate=5)  # capacity = 5
    buf.write(np.arange(1, 21, dtype=np.float32))  # 20 samples, only last 5 matter
    np.testing.assert_array_equal(buf.read(), [16, 17, 18, 19, 20])
    assert buf.is_full is True


def test_read_returns_a_copy_not_a_view():
    buf = AudioRingBuffer(capacity_seconds=1.0, sample_rate=5)
    buf.write(np.array([1, 2, 3], dtype=np.float32))
    snapshot = buf.read()
    snapshot[:] = 999.0
    np.testing.assert_array_equal(buf.read(), [1, 2, 3])  # unaffected by mutation above


def test_clear_resets_buffer_to_empty():
    buf = AudioRingBuffer(capacity_seconds=1.0, sample_rate=5)
    buf.write(np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))
    buf.clear()
    assert buf.filled_samples == 0
    assert buf.duration_seconds == 0.0
    np.testing.assert_array_equal(buf.read(), np.zeros(0, dtype=np.float32))


def test_write_ignores_empty_chunk():
    buf = AudioRingBuffer(capacity_seconds=1.0, sample_rate=5)
    buf.write(np.array([1, 2], dtype=np.float32))
    buf.write(np.zeros(0, dtype=np.float32))
    np.testing.assert_array_equal(buf.read(), [1, 2])


def test_wraparound_with_many_small_writes_preserves_recency_order():
    """Simulates many small streaming chunks wrapping the buffer repeatedly."""
    buf = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)  # capacity = 10
    for i in range(1, 26):  # write 25 single-sample chunks
        buf.write(np.array([float(i)], dtype=np.float32))
    # Last 10 values written were 16..25, in order.
    np.testing.assert_array_equal(buf.read(), np.arange(16, 26, dtype=np.float32))


def test_duration_seconds_reflects_filled_amount():
    buf = AudioRingBuffer(capacity_seconds=2.0, sample_rate=100)  # capacity = 200 samples
    buf.write(np.zeros(50, dtype=np.float32))
    assert buf.duration_seconds == pytest.approx(0.5)
    buf.write(np.zeros(200, dtype=np.float32))  # overflow capacity
    assert buf.duration_seconds == pytest.approx(2.0)
