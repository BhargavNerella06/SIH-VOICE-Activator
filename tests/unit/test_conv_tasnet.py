"""
tests/unit/test_conv_tasnet.py
===============================
Required tests (per project spec):
  1. Conv-TasNet can still be imported.
  2. The model can be instantiated.
  3. A test audio array can pass through the separation engine.
  4. The returned separated audio has the expected shape and type.
  5. (integration) FastAPI application starts successfully -> see test_api.py

Additional unit tests for the moved model code.
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# 1. Import test
# ---------------------------------------------------------------------------
def test_conv_tasnet_importable():
    """Conv-TasNet must be importable from the voice_separation package."""
    from voice_separation import ConvTasNet  # noqa: F401
    assert ConvTasNet is not None


# ---------------------------------------------------------------------------
# 2. Instantiation test (minimal hyper-params)
# ---------------------------------------------------------------------------
def test_conv_tasnet_instantiation():
    """The model must instantiate without errors with valid hyper-params."""
    from voice_separation.conv_tasnet import ConvTasNet

    model = ConvTasNet(
        N=64, L=16, B=64, H=128, P=3, X=4, R=2, C=2,
        norm_type="gLN", causal=False, mask_nonlinear="relu",
    )
    assert isinstance(model, torch.nn.Module)
    assert model.C == 2


# ---------------------------------------------------------------------------
# 3 & 4. Forward pass: shape and type
# ---------------------------------------------------------------------------
def test_conv_tasnet_forward_shape():
    """
    A synthetic mixture tensor must pass through the model and produce
    separated sources with the expected shape [M, C, T].
    """
    from voice_separation.conv_tasnet import ConvTasNet

    torch.manual_seed(0)
    N, L, B, H, P, X, R, C = 64, 16, 64, 128, 3, 4, 2, 2
    model = ConvTasNet(N, L, B, H, P, X, R, C, norm_type="gLN", causal=False)
    model.eval()

    M, T = 1, 8000  # batch=1, 1 second at 8 kHz
    mixture = torch.randn(M, T)

    with torch.no_grad():
        output = model(mixture)

    assert output.shape == (M, C, T), (
        f"Expected shape ({M}, {C}, {T}), got {tuple(output.shape)}"
    )
    assert output.dtype == torch.float32


# ---------------------------------------------------------------------------
# 5. SeparatorEngine: numpy in / numpy out
# ---------------------------------------------------------------------------
def test_separator_engine_numpy_io():
    """
    SeparatorEngine.separate() must accept a float32 numpy array and return
    a list of float32 numpy arrays, each of shape (T,).
    """
    from voice_separation.engine import SeparatorEngine

    # Instantiate without a checkpoint -> uses torchaudio bundle.
    # If torchaudio is not installed this test is skipped.
    pytest.importorskip("torchaudio", reason="torchaudio not installed")

    engine = SeparatorEngine(device="cpu")
    assert engine.is_loaded, f"Engine failed to load: {engine.status.error}"

    sr = 8000
    T = sr * 2  # 2 seconds
    waveform = np.random.randn(T).astype(np.float32)

    sources = engine.separate(waveform, sample_rate=sr)

    assert isinstance(sources, list), "separate() must return a list"
    assert len(sources) >= 1, "At least one source must be returned"
    for i, src in enumerate(sources):
        assert isinstance(src, np.ndarray), f"Source {i} is not ndarray"
        assert src.dtype == np.float32, f"Source {i} dtype is {src.dtype}"
        assert src.ndim == 1, f"Source {i} must be 1-D, got shape {src.shape}"
        assert src.shape[0] == T, (
            f"Source {i} length {src.shape[0]} != input length {T}"
        )


# ---------------------------------------------------------------------------
# 6. ConvTasNet deterministic behaviour
# ---------------------------------------------------------------------------
def test_conv_tasnet_deterministic():
    """Same input with eval() + no_grad must produce identical outputs."""
    from voice_separation.conv_tasnet import ConvTasNet

    torch.manual_seed(42)
    model = ConvTasNet(64, 16, 64, 128, 3, 4, 2, 2)
    model.eval()
    x = torch.randn(1, 8000)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.allclose(out1, out2), "Conv-TasNet output is not deterministic"


# ---------------------------------------------------------------------------
# 7. SeparatorEngine: NOT loaded without torchaudio and no checkpoint
# ---------------------------------------------------------------------------
def test_separator_engine_status_fields():
    """EngineStatus must have the correct fields when torchaudio is available."""
    pytest.importorskip("torchaudio", reason="torchaudio not installed")
    from voice_separation.engine import SeparatorEngine

    engine = SeparatorEngine(device="cpu")
    s = engine.status
    assert hasattr(s, "loaded")
    assert hasattr(s, "source")
    assert hasattr(s, "model_C")
