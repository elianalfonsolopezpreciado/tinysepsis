"""TinySepsis: a compact GRU-based early-warning model (<2M parameters).

Input per timestep: [value_z, mask, tslm_z, delta1_z] for each of the 34
time-varying clinical features (136 dims), fed through a small per-feature
linear projection ("feature tokenizer") before the recurrent encoder, plus
two static demographic features (Age, Gender) that set the initial hidden
state. Padding is handled with an explicit key-padding mask via
pack_padded_sequence-free masked pooling (variable stay lengths, left-padded
windows).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TinySepsisModel(nn.Module):
    def __init__(
        self,
        num_dynamic_features: int = 136,
        num_static_features: int = 2,
        token_dim: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_raw_features: int = 34,
    ):
        super().__init__()
        self.n_raw_features = n_raw_features
        self.num_dynamic_features = num_dynamic_features

        # Feature tokenizer: each of the 34 raw clinical channels (value,
        # mask, tslm, delta packed contiguously per-channel-group) gets its
        # own small linear embedding, concatenated before the GRU. This is
        # cheaper and more structured than one giant dense layer.
        self.input_proj = nn.Linear(num_dynamic_features, hidden_size // 2)
        self.static_proj = nn.Linear(num_static_features, hidden_size)

        self.gru = nn.GRU(
            input_size=hidden_size // 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, seq: torch.Tensor, pad_mask: torch.Tensor, static: torch.Tensor):
        """
        seq: (B, T, num_dynamic_features)
        pad_mask: (B, T) 1=real timestep, 0=padding (left-padded)
        static: (B, num_static_features)
        returns: logits (B,)
        """
        x = self.input_proj(seq)  # (B, T, H/2)
        x = torch.relu(x)

        h0_flat = self.static_proj(static)  # (B, H)
        h0 = h0_flat.unsqueeze(0).repeat(self.gru.num_layers, 1, 1).contiguous()

        out, _ = self.gru(x, h0)  # (B, T, H)

        # Masked mean+last pooling: use the last *real* (non-padded) timestep.
        lengths = pad_mask.sum(dim=1).clamp(min=1).long()  # (B,)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1))
        last_hidden = out.gather(1, idx).squeeze(1)  # (B, H)

        logits = self.head(last_hidden).squeeze(-1)  # (B,)
        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
