"""
Distillation losses for P4.

Teacher semantics
-----------------
EHGNN teacher outputs **raw logits** before ``log_softmax`` (see NC main.py).
``distill_loss()`` accepts ``teacher_logits`` as raw logits.

Student outputs raw logits; losses use ``F.log_softmax(student / T)`` vs
``F.softmax(teacher / T)`` for KL term.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_ce(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(student_logits, labels.long())


def kl_distillation(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL( student_T || teacher_T ) with batchmean reduction, scaled by T^2."""
    t = max(temperature, 1e-6)
    s_log = F.log_softmax(student_logits / t, dim=1)
    t_soft = F.softmax(teacher_logits / t, dim=1)
    return F.kl_div(s_log, t_soft, reduction="batchmean") * (t * t)


def mixed_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 3.0,
    kd_alpha: float = 0.5,
) -> torch.Tensor:
    """
    loss = (1 - kd_alpha) * CE + kd_alpha * T^2 * KL
    """
    ce = supervised_ce(student_logits, labels)
    kd = kl_distillation(student_logits, teacher_logits, temperature)
    return (1.0 - kd_alpha) * ce + kd_alpha * kd
