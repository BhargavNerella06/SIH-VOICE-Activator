"""
tests/unit/test_kws_model.py
===============================
Tests for keyword_spotting.model.TinyKWSNet.
"""

import torch


def test_tiny_kws_net_forward_shape():
    from keyword_spotting.model import TinyKWSNet
    model = TinyKWSNet(n_mels=40, num_classes=2)
    model.eval()
    x = torch.randn(2, 1, 40, 101)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2)
    assert out.dtype == torch.float32


def test_tiny_kws_net_handles_variable_time_dimension():
    """Adaptive pooling means the model must not care about exact T."""
    from keyword_spotting.model import TinyKWSNet
    model = TinyKWSNet(n_mels=40)
    model.eval()
    for T in (17, 101, 250):
        x = torch.randn(1, 1, 40, T)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)


def test_tiny_kws_net_is_lightweight():
    """Sized for eventual edge/MCU deployment -- must stay far smaller than Conv-TasNet's ~5M params."""
    from keyword_spotting.model import TinyKWSNet
    model = TinyKWSNet(n_mels=40)
    n_params = model.num_parameters()
    assert 0 < n_params < 50_000, f"expected a lightweight model, got {n_params} params"


def test_tiny_kws_net_deterministic_in_eval_mode():
    from keyword_spotting.model import TinyKWSNet
    torch.manual_seed(0)
    model = TinyKWSNet(n_mels=40)
    model.eval()
    x = torch.randn(1, 1, 40, 101)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)
