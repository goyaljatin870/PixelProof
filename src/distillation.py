"""
distillation.py - Cross-Modal Knowledge Distillation Engine
Compresses the high-capacity Teacher Ensemble (ResNet/EfficientNet + Wav2Vec2 + RoBERTa)
into the lightweight Edge Student Model (MobileNetV3 + Edge Heads) using soft-target KL-Divergence.

Loss Formula:
    L_KD = (1 - alpha) * H(y, sigma(z_s)) + alpha * (T^2) * D_KL(sigma(z_s / T) || sigma(z_t / T))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from src.config import config


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss with Temperature-Scaled KL-Divergence
    and Discrepancy Regularization.
    """
    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 2.0,
        discrepancy_weight: float = 0.2
    ):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.discrepancy_weight = discrepancy_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_disc_dict: Dict[str, torch.Tensor],
        teacher_disc_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the complete distillation loss.
        """
        # 1. Hard Cross-Entropy loss with ground truth labels
        loss_ce = self.ce_loss(student_logits, labels)

        # 2. Soft KL-Divergence loss between student and teacher softened probability distributions
        s_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        t_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        loss_kl = self.kl_loss(s_soft, t_soft) * (self.temperature ** 2)

        # 3. Discrepancy distillation: align student discrepancy representation with teacher
        loss_disc = 0.0
        for k in student_disc_dict:
            if k in teacher_disc_dict:
                loss_disc += F.mse_loss(student_disc_dict[k], teacher_disc_dict[k].detach())

        total_loss = ((1.0 - self.alpha) * loss_ce) + (self.alpha * loss_kl) + (self.discrepancy_weight * loss_disc)

        loss_breakdown = {
            "total_loss": float(total_loss.item()),
            "ce_loss": float(loss_ce.item()),
            "kl_loss": float(loss_kl.item()),
            "disc_loss": float(loss_disc.item() if isinstance(loss_disc, torch.Tensor) else loss_disc)
        }
        return total_loss, loss_breakdown
