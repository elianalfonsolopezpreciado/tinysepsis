"""TinySepsisCDE: a Neural Controlled Differential Equation (Kidger et al.,
NeurIPS 2020) alternative encoder for TinySepsis, implementing Priority 2
of regulatory/model_improvement_roadmap.md.

Motivation: the GRU encoder (tiny_sepsis.py) processes the same fixed,
hourly-binned, forward-filled+mask+delta representation every model in
this project uses. A Neural CDE instead treats the observed values as
control points on a continuous path X(t) (interpolated with a Hermite
cubic spline using only *backward* differences -- causal, no look-ahead
into future observations, which matters for a model meant to run online)
and integrates a learned vector field against that path. This is a more
literature-faithful way to handle irregular/missing observations than a
discretized RNN, and several 2024-2025 sepsis-prediction papers report
it improves discrimination on comparable ICU cohorts (see the roadmap
doc for citations). We test it here on the SAME 136-dim feature
representation (value/mask/tslm/delta) as the GRU, not a redesigned
input, so the comparison isolates "GRU vs. continuous-time integration"
rather than conflating it with a different missingness-handling scheme.

Trade-off made explicit: cdeint's adaptive ODE solver is slower than a
single GRU forward pass, and this model does not (yet) have an ONNX
export path -- torchdiffeq's adaptive solvers do not trace cleanly.
That's a real deployability cost this project's own thesis says should
be weighed against any accuracy gain, not ignored (see the roadmap
doc's "what NOT to do" section).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchcde


class _CDEFunc(nn.Module):
    """The learned vector field f_theta: R^hidden -> R^(hidden x input_channels)."""

    def __init__(self, hidden_size: int, input_channels: int, mlp_hidden: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_channels = input_channels
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, hidden_size * input_channels),
            nn.Tanh(),  # bounds the vector field, standard practice for CDE/ODE training stability
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(*z.shape[:-1], self.hidden_size, self.input_channels)


class TinySepsisCDEModel(nn.Module):
    def __init__(
        self,
        num_dynamic_features: int = 136,
        num_static_features: int = 2,
        hidden_size: int = 64,
        mlp_hidden: int = 64,
        dropout: float = 0.1,
        solver: str = "rk4",
        solver_options: dict | None = None,
    ):
        super().__init__()
        self.num_dynamic_features = num_dynamic_features
        self.hidden_size = hidden_size
        # +1 input channel for time itself, per torchcde convention.
        self.cde_func = _CDEFunc(hidden_size, num_dynamic_features + 1, mlp_hidden)
        self.static_proj = nn.Linear(num_static_features, hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        self.solver = solver
        # rk4 with a fixed step is much faster than an adaptive solver for
        # a short (24-step) sequence and avoids adaptive-step numerical
        # warnings on the bounded, already-normalized inputs this model sees.
        self.solver_options = solver_options or {"step_size": 1.0}

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """
        seq: (B, T, num_dynamic_features) -- same left-padded, hourly-gridded
             representation as TinySepsisModel.
        static: (B, num_static_features)
        returns: logits (B,)
        """
        B, T, _ = seq.shape
        t = torch.arange(T, dtype=seq.dtype, device=seq.device).view(1, T, 1).expand(B, T, 1)
        path = torch.cat([t, seq], dim=-1)  # (B, T, num_dynamic_features + 1)

        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(path)
        X = torchcde.CubicSpline(coeffs)

        z0 = self.static_proj(static)  # (B, hidden_size)
        zt = torchcde.cdeint(
            X=X, func=self.cde_func, z0=z0, t=X.interval,
            method=self.solver, options=self.solver_options, adjoint=False,
        )  # (B, 2, hidden_size): [t_start, t_end]
        z_final = zt[:, -1]  # state at the most recent (left-padded-last) observation

        logits = self.head(z_final).squeeze(-1)
        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
