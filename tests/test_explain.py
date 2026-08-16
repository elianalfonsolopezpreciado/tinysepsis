import numpy as np
import torch

from tinysepsis.eval.explain import aggregate_feature_attributions, integrated_gradients
from tinysepsis.models.tiny_sepsis import TinySepsisModel


def _small_model():
    torch.manual_seed(0)
    model = TinySepsisModel(num_dynamic_features=16, num_static_features=2, token_dim=4, hidden_size=16, num_layers=1)
    model.eval()
    return model


def test_completeness_axiom():
    """The defining correctness property of Integrated Gradients: attributions
    must sum to model(input) - model(baseline), not just look plausible."""
    model = _small_model()
    seq = torch.randn(1, 6, 16)
    static = torch.randn(1, 2)
    baseline_seq = torch.zeros_like(seq)
    baseline_static = torch.zeros_like(static)

    seq_attr, static_attr = integrated_gradients(model, seq, static, steps=200)

    with torch.no_grad():
        f_input = model(seq, static).item()
        f_baseline = model(baseline_seq, baseline_static).item()

    total_attr = seq_attr.sum().item() + static_attr.sum().item()
    assert abs(total_attr - (f_input - f_baseline)) < 1e-2


def test_zero_attribution_when_input_equals_baseline():
    model = _small_model()
    seq = torch.randn(1, 6, 16)
    static = torch.randn(1, 2)
    seq_attr, static_attr = integrated_gradients(model, seq, static, baseline_seq=seq, baseline_static=static, steps=10)
    assert torch.allclose(seq_attr, torch.zeros_like(seq_attr), atol=1e-6)
    assert torch.allclose(static_attr, torch.zeros_like(static_attr), atol=1e-6)


def test_attribution_shapes():
    model = _small_model()
    seq = torch.randn(3, 6, 16)
    static = torch.randn(3, 2)
    seq_attr, static_attr = integrated_gradients(model, seq, static, steps=8)
    assert seq_attr.shape == seq.shape
    assert static_attr.shape == static.shape


def test_aggregate_feature_attributions_shape_and_sign():
    T, n_raw = 4, 4
    seq_attr = np.zeros((T, 4 * n_raw), dtype=np.float32)
    seq_attr[0, 0] = 3.0  # value channel, feature 0, hour 0
    seq_attr[2, n_raw + 0] = -2.0  # mask channel, feature 0, hour 2 -- should add via abs()
    static_attr = np.array([0.5, -0.1], dtype=np.float32)

    result = aggregate_feature_attributions(
        seq_attr, static_attr, feature_names=["A", "B", "C", "D"], static_names=["Age", "Gender"], n_raw_features=n_raw
    )
    assert set(result.keys()) == {"A", "B", "C", "D", "Age", "Gender"}
    assert abs(result["A"] - 5.0) < 1e-5  # |3.0| + |-2.0|
    assert result["B"] == 0.0
    assert abs(result["Age"] - 0.5) < 1e-5
    assert abs(result["Gender"] - 0.1) < 1e-5


def test_aggregate_rejects_wrong_channel_count():
    seq_attr = np.zeros((4, 15), dtype=np.float32)  # not a multiple of 4
    static_attr = np.zeros((2,), dtype=np.float32)
    try:
        aggregate_feature_attributions(seq_attr, static_attr, feature_names=["A"] * 4, static_names=["Age", "Gender"], n_raw_features=4)
        assert False, "expected AssertionError"
    except AssertionError:
        pass
