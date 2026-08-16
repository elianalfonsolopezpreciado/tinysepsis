import torch

from tinysepsis.models.tiny_sepsis_cde import TinySepsisCDEModel


def test_forward_output_shape():
    model = TinySepsisCDEModel(num_dynamic_features=16, num_static_features=2, hidden_size=8, mlp_hidden=16)
    batch, seq_len = 4, 12
    seq = torch.randn(batch, seq_len, 16)
    static = torch.randn(batch, 2)
    logits = model(seq, static)
    assert logits.shape == (batch,)
    assert torch.isfinite(logits).all()


def test_gradients_flow_to_all_parameters():
    model = TinySepsisCDEModel(num_dynamic_features=8, num_static_features=2, hidden_size=8, mlp_hidden=16)
    seq = torch.randn(2, 6, 8)
    static = torch.randn(2, 2)
    logits = model(seq, static)
    loss = logits.sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_parameter_count_reasonable():
    model = TinySepsisCDEModel(num_dynamic_features=136, num_static_features=2, hidden_size=64, mlp_hidden=64)
    n = model.num_parameters()
    assert 0 < n < 10_000_000


def test_constant_sequence_is_a_fixed_point_of_a_zero_vector_field():
    """Sanity check on the CDE mechanics themselves, independent of learned
    weights: if the vector field is exactly zero, integrating against any
    path leaves the hidden state unchanged (dz = f(z) dX = 0 dX = 0)."""
    model = TinySepsisCDEModel(num_dynamic_features=4, num_static_features=2, hidden_size=4, mlp_hidden=8)
    model.eval()  # disable dropout stochasticity so the two head() calls below are directly comparable
    with torch.no_grad():
        for p in model.cde_func.net.parameters():
            p.zero_()
    seq = torch.randn(1, 6, 4)
    static = torch.randn(1, 2)
    with torch.no_grad():
        logits = model(seq, static)
        expected = model.head(model.static_proj(static)).squeeze(-1)
    assert torch.allclose(logits, expected, atol=1e-3)
