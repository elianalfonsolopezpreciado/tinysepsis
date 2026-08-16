"""TinySepsisAttn: a lightweight self-attention encoder alternative to the
GRU (tiny_sepsis.py), implementing Tier 2 item 4 of
regulatory/model_improvement_roadmap.md.

This uses PyTorch's standard nn.TransformerEncoder (multi-head
scaled-dot-product self-attention), not the literal kernel-attention
mechanism of the KA-Transformer paper the roadmap cites -- that paper's
specific contribution is a linear-time attention approximation, which is
an implementation-efficiency concern more than an accuracy one, and adding
a hand-rolled kernel-attention implementation would add real bug surface
for a benefit orthogonal to what this comparison is actually testing:
whether an attention-based encoder, sized to the same small-parameter
budget as the GRU, changes discrimination or cross-institution robustness
at all. Standard attention answers that question equally well at this
sequence length (T=24, quadratic attention here is 24x24, trivially cheap)
and is far less likely to be a source of new bugs.

Design mirrors the GRU as closely as possible so the comparison isolates
"recurrence vs. attention," not confounded by unrelated architecture
choices: same 136-dim per-timestep feature encoding, same left-padded
convention (the current/most recent observation is always at position
T-1), same final-position pooling, same head shape. Static features
condition the encoder by being added to every token's embedding (the
attention analogue of the GRU's static-conditioned initial hidden state
h0), rather than introducing a separate CLS-token mechanism.

No explicit padding/causal attention mask: like the GRU (which also
receives no pad_mask -- see tiny_sepsis.py), the model is given the
mask/time-since-last-measurement channels as part of each token's own
features, which is sufficient signal for it to learn to down-weight
zero-padded early positions itself, and keeps this model exportable to
ONNX without the added complexity of a variable-length mask input.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TinySepsisAttnModel(nn.Module):
    def __init__(
        self,
        num_dynamic_features: int = 136,
        num_static_features: int = 2,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        seq_len: int = 24,
    ):
        super().__init__()
        self.num_dynamic_features = num_dynamic_features
        self.d_model = d_model

        self.input_proj = nn.Linear(num_dynamic_features, d_model)
        self.static_proj = nn.Linear(num_static_features, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """
        seq: (B, T, num_dynamic_features), left-padded (see tiny_sepsis.py's
             forward() docstring -- the same convention applies here).
        static: (B, num_static_features)
        returns: logits (B,)
        """
        B, T, _ = seq.shape
        x = self.input_proj(seq) + self.pos_embedding[:, :T, :]
        static_tok = self.static_proj(static).unsqueeze(1)  # (B, 1, d_model)
        x = x + static_tok  # broadcast: condition every position on demographics, GRU-h0 analogue

        out = self.encoder(x)  # (B, T, d_model)
        last_hidden = out[:, -1, :]  # most recent real observation, per the left-padded convention

        logits = self.head(last_hidden).squeeze(-1)
        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
