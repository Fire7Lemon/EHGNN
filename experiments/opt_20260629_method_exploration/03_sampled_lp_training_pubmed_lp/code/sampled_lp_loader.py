"""
Sampled-LP training utilities for P3.

Sampling object (matches Link Prediction/main.py structure)
------------------------------------------------------------
Original PubMed LP builds a 1-D tensor ``train_idx`` of length ``num_sample``
(= number of disease nodes) each epoch:

    train_idx = randint(0, train_links.shape[0], (num_sample,))

Each value is an **index into the positive train-link rows** used as
``train_links[0][batch_train]`` / ``train_links[1][batch_train]`` inside
DataLoader mini-batches.  One optimizer step = one batch of these indices;
positive/negative pairs per step stay 1:1 via ``neg_sample``.

P3 scales **how many train indices are drawn per epoch**:

    n_steps = round(num_sample * sample_ratio)

When ``sample_ratio == 1.0``, ``n_steps == num_sample`` (full-training mode).

This reduces training steps per epoch while keeping:
- original dot-product decoder
- original BCELoss(sigmoid(dot))
- full-graph ``evaluate_lp()`` unchanged
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SamplingProtocol:
    """Human-readable description of the sampling setup."""

    sample_ratio: float
    resample_each_epoch: bool
    base_steps: int
    steps_per_epoch: int
    index_upper: int
    sampling_object: str = "train_index_tensor_per_epoch"
    keeps_pos_neg_ratio: bool = True
    evaluation: str = "full evaluate_lp() unchanged"


def compute_steps_per_epoch(base_steps: int, sample_ratio: float) -> int:
    if sample_ratio <= 0:
        raise ValueError("sample_ratio must be > 0")
    if sample_ratio >= 1.0:
        return base_steps
    return max(1, int(round(base_steps * sample_ratio)))


def describe_protocol(
    base_steps: int,
    sample_ratio: float,
    resample_each_epoch: bool,
    index_upper: int,
) -> SamplingProtocol:
    steps = compute_steps_per_epoch(base_steps, sample_ratio)
    return SamplingProtocol(
        sample_ratio=sample_ratio,
        resample_each_epoch=resample_each_epoch,
        base_steps=base_steps,
        steps_per_epoch=steps,
        index_upper=index_upper,
    )


def sample_train_indices(
    steps_per_epoch: int,
    index_upper: int,
    resample_each_epoch: bool,
    epoch: int,
    base_seed: int,
    fixed_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Return ``train_idx`` tensor of shape ``[steps_per_epoch]`` for one epoch.

    Parameters
    ----------
    index_upper : upper bound for randint (``train_links.shape[0]`` in main.py)
    fixed_indices : reused when ``resample_each_epoch=False``
    """
    if not resample_each_epoch and fixed_indices is not None:
        return fixed_indices, fixed_indices

    rng = np.random.default_rng(base_seed + (epoch if resample_each_epoch else 0))
    idx = rng.integers(0, index_upper, size=steps_per_epoch, dtype=np.int64)
    tensor = torch.from_numpy(idx).long()
    if not resample_each_epoch:
        return tensor, tensor
    return tensor, fixed_indices


def init_fixed_indices(
    steps_per_epoch: int,
    index_upper: int,
    base_seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(base_seed)
    idx = rng.integers(0, index_upper, size=steps_per_epoch, dtype=np.int64)
    return torch.from_numpy(idx).long()


def ratio_tag(sample_ratio: float) -> str:
    """Filename tag, e.g. 0.5 -> ratio050, 0.25 -> ratio025."""
    pct = int(round(sample_ratio * 100))
    return "ratio{:03d}".format(pct)
