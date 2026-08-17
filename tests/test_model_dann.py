import torch

from tinysepsis.models.tiny_sepsis_dann import GradientReversal, TinySepsisDANNModel, grad_reverse


def test_gradient_reversal_forward_is_identity():
    x = torch.randn(5, 3)
    out = grad_reverse(x, lambd=1.0)
    assert torch.equal(out, x)


def test_gradient_reversal_negates_and_scales_gradient():
    x = torch.randn(4, requires_grad=True)
    y = grad_reverse(x, lambd=2.5)
    loss = y.sum()
    loss.backward()
    # d(sum)/dx would normally be all-ones; GRL must flip sign and scale by lambd.
    assert torch.allclose(x.grad, torch.full_like(x, -2.5))


def test_forward_output_shapes():
    model = TinySepsisDANNModel(num_dynamic_features=16, num_static_features=2, hidden_size=8, num_layers=1)
    batch, seq_len = 6, 12
    seq = torch.randn(batch, seq_len, 16)
    static = torch.randn(batch, 2)
    task_logits, domain_logits = model(seq, static, lambd=1.0)
    assert task_logits.shape == (batch,)
    assert domain_logits.shape == (batch,)
    assert torch.isfinite(task_logits).all() and torch.isfinite(domain_logits).all()


def test_gradients_flow_to_encoder_from_both_heads():
    model = TinySepsisDANNModel(num_dynamic_features=8, num_static_features=2, hidden_size=8, num_layers=1)
    seq = torch.randn(3, 6, 8)
    static = torch.randn(3, 2)
    task_logits, domain_logits = model(seq, static, lambd=1.0)
    (task_logits.sum() + domain_logits.sum()).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_lambd_zero_still_computes_domain_logits_but_no_encoder_pressure():
    """lambd=0 should not error, and should make the encoder's gradient
    contribution from the domain head exactly zero (pure forward pass
    through the domain head with no adversarial signal)."""
    model = TinySepsisDANNModel(num_dynamic_features=8, num_static_features=2, hidden_size=8, num_layers=1)
    seq = torch.randn(3, 6, 8)
    static = torch.randn(3, 2)
    _, domain_logits = model(seq, static, lambd=0.0)
    domain_logits.sum().backward()
    assert torch.allclose(model.input_proj.weight.grad, torch.zeros_like(model.input_proj.weight.grad))
