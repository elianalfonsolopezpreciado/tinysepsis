import torch

from tinysepsis.models.tiny_sepsis_attn import TinySepsisAttnModel


def test_forward_output_shape():
    model = TinySepsisAttnModel(num_dynamic_features=16, num_static_features=2, d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
    batch, seq_len = 4, 12
    seq = torch.randn(batch, seq_len, 16)
    static = torch.randn(batch, 2)
    logits = model(seq, static)
    assert logits.shape == (batch,)
    assert torch.isfinite(logits).all()


def test_gradients_flow_to_all_parameters():
    model = TinySepsisAttnModel(num_dynamic_features=8, num_static_features=2, d_model=8, nhead=2, num_layers=1, dim_feedforward=16, seq_len=6)
    seq = torch.randn(2, 6, 8)
    static = torch.randn(2, 2)
    logits = model(seq, static)
    loss = logits.sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_parameter_count_under_gru_budget():
    """Tier 2 item 4's whole premise is a parameter-efficient alternative to
    the GRU -- this model should stay comfortably under the GRU's <200K
    budget at its default (production-candidate) size, not just be small in
    the tiny test configurations used above."""
    model = TinySepsisAttnModel(num_dynamic_features=136, num_static_features=2)
    n = model.num_parameters()
    assert 0 < n < 200_000


def test_last_position_pooling_matches_manual_reference():
    """Regression guard for the same class of bug tiny_sepsis.py's forward()
    docstring warns about: pooling must read position T-1 (the most recent
    real observation under this project's left-padding convention), not
    something length-dependent."""
    torch.manual_seed(0)
    model = TinySepsisAttnModel(num_dynamic_features=4, num_static_features=2, d_model=8, nhead=2, num_layers=1, dim_feedforward=16, seq_len=5)
    model.eval()
    seq = torch.randn(1, 5, 4)
    static = torch.randn(1, 2)

    with torch.no_grad():
        x = model.input_proj(seq) + model.pos_embedding[:, :5, :]
        x = x + model.static_proj(static).unsqueeze(1)
        out = model.encoder(x)
        expected_logit = model.head(out[:, -1, :]).squeeze(-1)
        actual_logit = model(seq, static)

    assert torch.allclose(actual_logit, expected_logit, atol=1e-6)
