"""
tests/unit/test_engine_caching.py
==================================
Verify that SeparatorEngine is loaded once per process and reused,
instead of being reconstructed on every call.
"""

import pytest


@pytest.fixture(autouse=True)
def _clear_engine_cache():
    """Isolate each test from cache state left by other tests."""
    from voice_separation.engine import clear_engine_cache
    clear_engine_cache()
    yield
    clear_engine_cache()


def test_get_separator_engine_returns_same_instance():
    """Two calls with the same args must return the identical cached object."""
    pytest.importorskip("torchaudio", reason="torchaudio not installed")
    from voice_separation.engine import get_separator_engine

    e1 = get_separator_engine()
    e2 = get_separator_engine()
    assert e1 is e2


def test_get_separator_engine_loads_model_only_once(monkeypatch):
    """The underlying model-loading routine must run exactly once, not per call."""
    pytest.importorskip("torchaudio", reason="torchaudio not installed")
    import voice_separation.engine as engine_mod

    call_count = {"n": 0}
    original = engine_mod.SeparatorEngine._load_torchaudio_bundle

    def counting_load(self):
        call_count["n"] += 1
        return original(self)

    monkeypatch.setattr(
        engine_mod.SeparatorEngine, "_load_torchaudio_bundle", counting_load
    )

    engine_mod.get_separator_engine()
    engine_mod.get_separator_engine()
    engine_mod.get_separator_engine()

    assert call_count["n"] == 1, (
        f"Model bundle was loaded {call_count['n']} times; expected exactly once"
    )


def test_get_separator_engine_is_loaded_and_correct_defaults():
    pytest.importorskip("torchaudio", reason="torchaudio not installed")
    from voice_separation.engine import get_separator_engine

    engine = get_separator_engine()
    assert engine.is_loaded
    assert engine.status.source == "torchaudio_bundle"
    assert engine.status.model_C == 2
    assert engine.sample_rate == 8000
