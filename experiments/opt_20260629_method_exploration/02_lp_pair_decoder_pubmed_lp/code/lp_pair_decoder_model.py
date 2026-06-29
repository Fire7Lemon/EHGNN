"""
Link-prediction decoders for P2 Pair Decoder experiment.

All decoders accept ``src_emb`` and ``dst_emb`` with shape ``[batch, emb_dim]``.

Output semantics
----------------
- ``forward(..., return_logits=True)``  -> raw logit ``[batch]`` (use with BCEWithLogitsLoss)
- ``forward(..., return_logits=False)`` -> probability ``sigmoid(logit)`` (use with utils.accuracy)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DotDecoder(nn.Module):
    """Baseline dot-product decoder: logit = sum(h_u * h_v, dim=-1)."""

    def forward(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        return_logits: bool = True,
    ) -> torch.Tensor:
        logits = (src_emb * dst_emb).sum(dim=-1)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


class PairMLPDecoder(nn.Module):
    """
    Pair-wise MLP decoder (SEAL/BUDDY-style features).

    z = [h_u, h_v, |h_u - h_v|, h_u * h_v] -> MLP -> logit
    """

    def __init__(self, emb_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.emb_dim = emb_dim
        in_dim = emb_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim >= 2 else 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2 if hidden_dim >= 2 else 1, 1),
        )

    def pair_features(self, src_emb: torch.Tensor, dst_emb: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(src_emb - dst_emb)
        prod = src_emb * dst_emb
        return torch.cat([src_emb, dst_emb, diff, prod], dim=-1)

    def forward(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        return_logits: bool = True,
    ) -> torch.Tensor:
        z = self.pair_features(src_emb, dst_emb)
        logits = self.mlp(z).squeeze(-1)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


def build_decoder(name: str, emb_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
    name = name.lower()
    if name == "dot":
        return DotDecoder()
    if name in ("pair_mlp", "pair-mlp", "pairmlp"):
        return PairMLPDecoder(emb_dim, hidden_dim, dropout)
    raise ValueError("Unknown decoder {!r}; use 'dot' or 'pair_mlp'".format(name))
