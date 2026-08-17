"""TinySepsisDANN: unsupervised domain-adversarial adaptation (Ganin &
Lempitsky, 2015, "Domain-Adversarial Training of Neural Networks") applied
to the cross-institution gap that is this project's central finding.

Motivation: every other model-improvement experiment in
regulatory/model_improvement_roadmap.md changes the architecture or
training recipe while still only ever seeing Hospital A during training --
Hospital B is used purely for blind evaluation. DANN is a different kind
of lever: it uses Hospital B's INPUT FEATURES (vitals/labs), never its
sepsis labels, during training, via a domain classifier trained
adversarially (through a gradient-reversal layer) against the shared GRU
encoder. The encoder is pushed to produce hidden representations a domain
classifier cannot tell apart between hospitals, while the task head keeps
learning to predict sepsis from Hospital A's labels through that same
representation -- the hope being that representations invariant to
hospital identity are also more transferable for the sepsis task itself.
Recent clinical-domain-generalization work reports this same idea (a
"lightweight adversarial framework to suppress hospital-specific cues
while preserving disease information") as effective in a similar setting.

IMPORTANT METHODOLOGICAL CAVEAT, not to be glossed over: because this
touches Hospital B's features (not labels) during training, evaluating the
result on Hospital B afterward is NOT the same genuinely-blind external
validation every other number in this project reports -- it is standard,
legitimate *transductive* unsupervised domain adaptation (a realistic
deployment scenario: a new hospital's unlabeled data stream is available
before any outcome labels are), but it is a methodologically different
claim and must always be reported as such, never conflated with the
blind-holdout numbers elsewhere.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, lambd)


class TinySepsisDANNModel(nn.Module):
    def __init__(
        self,
        num_dynamic_features: int = 136,
        num_static_features: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        domain_hidden: int = 32,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_dynamic_features, hidden_size // 2)
        self.static_proj = nn.Linear(num_static_features, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size // 2, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.task_head = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size // 2, 1),
        )
        # Domain classifier: predicts which hospital a representation came
        # from. Deliberately small/shallow -- a weak adversary that still
        # forces invariance is preferable to a strong one that just
        # overfits the domain-classification task instead of pressuring
        # the shared encoder.
        self.domain_head = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, domain_hidden),
            nn.GELU(), nn.Linear(domain_hidden, 1),
        )

    def encode(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_proj(seq))
        h0 = self.static_proj(static).unsqueeze(0).repeat(self.gru.num_layers, 1, 1).contiguous()
        out, _ = self.gru(x, h0)
        return out[:, -1, :]

    def forward(self, seq: torch.Tensor, static: torch.Tensor, lambd: float = 0.0):
        """Returns (task_logits, domain_logits). lambd=0 disables the
        gradient-reversal effect (pure forward pass, e.g. for plain
        inference) without disabling the domain head's own gradient to
        itself -- set lambd=0 and simply ignore domain_logits at eval time."""
        feat = self.encode(seq, static)
        task_logits = self.task_head(feat).squeeze(-1)
        domain_logits = self.domain_head(grad_reverse(feat, lambd)).squeeze(-1)
        return task_logits, domain_logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
