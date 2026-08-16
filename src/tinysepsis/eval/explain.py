"""Integrated Gradients (Sundararajan et al., 2017) attribution for
TinySepsisModel, replacing the previous "top contributing factors" proxy
in the demo/CDS Hooks integration, which was literally `abs(z-score of the
raw last-observed value)` -- a measure of how unusual a value looks, not
of what the model actually weighs. That proxy could rank a value the model
barely uses above one driving the prediction, which is actively misleading
in a clinical-decision-support context, not just imprecise.

This is a from-scratch implementation (no captum dependency) so the exact
integration rule stays inspectable -- appropriate for a component whose
whole purpose is regulatory/clinical defensibility (GMLP Principle 6:
human factors considerations, including the basis a clinician needs to
independently review a recommendation, per the Cures Act non-device CDS
framing in regulatory/intended_use_statement.md).
"""
from __future__ import annotations

import numpy as np
import torch


def integrated_gradients(model, seq, static, baseline_seq=None, baseline_static=None, steps=50):
    """Attribute model(seq, static) - model(baseline_seq, baseline_static) to
    each element of seq and static, via the path integral of gradients along
    the straight line from baseline to input, approximated with a right
    Riemann sum over `steps` points -- the standard IG estimator.

    Default baseline is all-zeros: for this model's z-scored/masked input
    representation, that is a "patient with no measurements ever taken"
    (mask=0 everywhere, value/time-since-last-measurement/delta channels at
    their neutral encoded value) and static features at [mean age, gender=0]
    -- a defensible, if not unique, reference point. The gender=0 baseline
    in particular is an arbitrary reference (not a claim that one gender is
    "neutral"); document this if attributions are ever surfaced with the
    static-feature breakdown shown to a clinician.

    model must be in eval() mode by the caller if it has dropout/batchnorm --
    IG's correctness relies on the interpolated points along the path all
    being evaluated by the same deterministic function; a stochastic model
    breaks the fundamental-theorem-of-calculus identity the method rests on.

    Returns (seq_attr, static_attr): tensors the same shape as seq/static.
    """
    if baseline_seq is None:
        baseline_seq = torch.zeros_like(seq)
    if baseline_static is None:
        baseline_static = torch.zeros_like(static)

    B = seq.shape[0]
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=seq.device)

    seq_diff = seq - baseline_seq
    static_diff = static - baseline_static

    interp_seq = baseline_seq.unsqueeze(0) + alphas.view(-1, 1, 1, 1) * seq_diff.unsqueeze(0)
    interp_static = baseline_static.unsqueeze(0) + alphas.view(-1, 1, 1) * static_diff.unsqueeze(0)
    interp_seq = interp_seq.reshape(steps * B, *seq.shape[1:]).clone().detach().requires_grad_(True)
    interp_static = interp_static.reshape(steps * B, *static.shape[1:]).clone().detach().requires_grad_(True)

    logits = model(interp_seq, interp_static)
    grads_seq, grads_static = torch.autograd.grad(logits.sum(), [interp_seq, interp_static])

    grads_seq = grads_seq.reshape(steps, B, *seq.shape[1:]).mean(dim=0)
    grads_static = grads_static.reshape(steps, B, *static.shape[1:]).mean(dim=0)

    seq_attr = (seq_diff * grads_seq).detach()
    static_attr = (static_diff * grads_static).detach()
    return seq_attr, static_attr


def aggregate_feature_attributions(seq_attr, static_attr, feature_names, static_names, n_raw_features=34):
    """Collapse a single example's (T, 4*n_raw_features) sequence attribution
    (channel blocks: [value, mask, tslm, delta], each n_raw_features wide,
    per tinysepsis.data.dataset's encoding) and (n_static,) static
    attribution into one dict[feature_name] -> total |attribution|, summed
    across the 4 channels and all T timesteps for the dynamic features.
    Ranking by absolute value (not signed sum) surfaces what the model is
    sensitive to, positively or negatively, which is what "what should a
    clinician double-check" needs -- a large negative attribution (e.g. a
    reassuringly normal value suppressing risk) is just as worth showing as
    a large positive one.
    """
    if seq_attr.ndim == 3:
        seq_attr = seq_attr[0]
    if static_attr.ndim == 2:
        static_attr = static_attr[0]

    seq_attr = seq_attr.detach().cpu().numpy() if isinstance(seq_attr, torch.Tensor) else np.asarray(seq_attr)
    static_attr = static_attr.detach().cpu().numpy() if isinstance(static_attr, torch.Tensor) else np.asarray(static_attr)

    T, F = seq_attr.shape
    assert F == 4 * n_raw_features, f"expected {4 * n_raw_features} dynamic channels, got {F}"

    per_channel = np.abs(seq_attr).reshape(T, 4, n_raw_features)
    per_feature = per_channel.sum(axis=(0, 1))  # (n_raw_features,)

    result = {name: float(per_feature[i]) for i, name in enumerate(feature_names)}
    for i, name in enumerate(static_names):
        result[name] = float(abs(static_attr[i]))
    return result
