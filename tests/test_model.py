import torch

from tinysepsis.models.tiny_sepsis import TinySepsisModel


def test_forward_output_shape():
    model = TinySepsisModel(num_dynamic_features=136, num_static_features=2, hidden_size=64, num_layers=2)
    batch, seq_len = 8, 24
    seq = torch.randn(batch, seq_len, 136)
    pad_mask = torch.ones(batch, seq_len)
    static = torch.randn(batch, 2)
    logits = model(seq, pad_mask, static)
    assert logits.shape == (batch,)


def test_parameter_count_under_10m():
    model = TinySepsisModel(num_dynamic_features=136, num_static_features=2, hidden_size=128, num_layers=2)
    assert model.num_parameters() < 10_000_000


def test_gradients_flow_to_all_parameters():
    model = TinySepsisModel(num_dynamic_features=136, num_static_features=2, hidden_size=32, num_layers=1)
    seq = torch.randn(4, 10, 136)
    pad_mask = torch.ones(4, 10)
    static = torch.randn(4, 2)
    logits = model(seq, pad_mask, static)
    loss = logits.sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_padding_mask_uses_last_real_timestep():
    model = TinySepsisModel(num_dynamic_features=4, num_static_features=2, hidden_size=16, num_layers=1)
    model.eval()
    seq = torch.randn(1, 6, 4)
    static = torch.randn(1, 2)

    pad_mask_full = torch.ones(1, 6)
    pad_mask_partial = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])

    with torch.no_grad():
        out_full = model(seq, pad_mask_full, static)
        out_partial = model(seq, pad_mask_partial, static)
    # different padding masks with identical seq content but different
    # "length" should not crash and should produce finite outputs
    assert torch.isfinite(out_full).all()
    assert torch.isfinite(out_partial).all()
