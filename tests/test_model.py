import torch

from tinysepsis.models.tiny_sepsis import TinySepsisModel


def test_forward_output_shape():
    model = TinySepsisModel(num_dynamic_features=136, num_static_features=2, hidden_size=64, num_layers=2)
    batch, seq_len = 8, 24
    seq = torch.randn(batch, seq_len, 136)
    static = torch.randn(batch, 2)
    logits = model(seq, static)
    assert logits.shape == (batch,)


def test_parameter_count_under_10m():
    model = TinySepsisModel(num_dynamic_features=136, num_static_features=2, hidden_size=128, num_layers=2)
    assert model.num_parameters() < 10_000_000


def test_gradients_flow_to_all_parameters():
    model = TinySepsisModel(num_dynamic_features=136, num_static_features=2, hidden_size=32, num_layers=1)
    seq = torch.randn(4, 10, 136)
    static = torch.randn(4, 2)
    logits = model(seq, static)
    loss = logits.sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_prediction_reads_the_true_last_timestep():
    """Regression test for a real bug that shipped for a period: an
    earlier implementation took a `pad_mask` argument and pooled at index
    `pad_mask.sum(dim=1) - 1`, which is only correct for RIGHT-padded
    sequences. This codebase's sequences are always LEFT-padded (see
    tinysepsis.data.dataset: the current hour's observation is placed at
    the final timestep T-1, regardless of how many real hours precede
    it), so that gather silently read a zero-padded position instead of
    the true last observation for any patient with fewer than T real
    hours -- the common case. The fix removes pad_mask from the model
    entirely and always pools at position -1. This test verifies the
    model does exactly that, against a hand-computed reference, rather
    than relying on a trained/random network being numerically
    "sensitive enough" to an input perturbation (a fragile thing to
    assert under random initialization).
    """
    torch.manual_seed(0)
    model = TinySepsisModel(num_dynamic_features=4, num_static_features=2, hidden_size=16, num_layers=1)
    model.eval()

    seq = torch.randn(2, 24, 4)
    static = torch.randn(2, 2)

    with torch.no_grad():
        actual_logits = model(seq, static)

        # Hand-reproduce the encoder up to the GRU output, then pool at
        # the true last position (T-1) -- this is what forward() must do.
        x = torch.relu(model.input_proj(seq))
        h0 = model.static_proj(static).unsqueeze(0).repeat(model.gru.num_layers, 1, 1).contiguous()
        gru_out, _ = model.gru(x, h0)
        expected_logits = model.head(gru_out[:, -1, :]).squeeze(-1)

    assert torch.allclose(actual_logits, expected_logits, atol=1e-6), (
        "model output does not match pooling at the true last timestep (T-1)"
    )


def test_prediction_is_sensitive_to_the_last_timestep_content():
    """A change to the final (most recent) timestep must change the
    prediction -- the direct, end-to-end version of the bug this project
    actually hit (see tests/test_cds_hooks.py for the full CDS-Hooks-level
    version): a deteriorating patient and a stable one must not collapse
    to the same score just because both have a short real history.
    """
    torch.manual_seed(1)
    model = TinySepsisModel(num_dynamic_features=4, num_static_features=2, hidden_size=16, num_layers=1)
    model.eval()

    seq = torch.zeros(1, 24, 4)
    static = torch.randn(1, 2)

    seq_extreme = seq.clone()
    seq_extreme[0, -1] = torch.tensor([8.0, -8.0, 8.0, -8.0])

    with torch.no_grad():
        out_baseline = model(seq, static)
        out_extreme = model(seq_extreme, static)

    assert not torch.allclose(out_baseline, out_extreme, atol=1e-3)
