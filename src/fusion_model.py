"""
fusion_model.py - Late-Fusion MLP & Inter-Modality Discrepancy Engine
Combines visual (V_f), acoustic (A_f), and textual (T_f) embeddings through
layer normalization, cross-attention discrepancy scoring, and a 3-layer residual MLP head.
Produces probability distribution across: Authentic, Manipulated Payload, Contextual Misinformation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional

from src.visual_branch import TeacherVisualBackbone, StudentVisualBackbone
from src.audio_branch import AcousticVocoderBackbone
from src.nlp_branch import IntentClaimBackbone
from src.config import config


class MultimodalLateFusionHead(nn.Module):
    """
    Late-Fusion Multilayer Perceptron & Inter-Modality Discrepancy Engine.
    
    Formula:
        F_concatenated = [ LayerNorm(V_f) || LayerNorm(A_f) || LayerNorm(T_f) ]
        Discrepancy Matrix: [ (V_f x A_f^T), (V_f x T_f^T), (A_f x T_f^T) ]
    """
    def __init__(
        self,
        visual_dim: int = 2048,
        audio_dim: int = 768,
        text_dim: int = 768,
        proj_dim: int = 256,
        hidden_dims: list = [512, 256, 128],
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.proj_dim = proj_dim
        self.num_classes = num_classes

        # Modality Layer Normalization
        self.ln_v = nn.LayerNorm(visual_dim)
        self.ln_a = nn.LayerNorm(audio_dim)
        self.ln_t = nn.LayerNorm(text_dim)

        # Projections into shared latent space for discrepancy cross-multiplication
        self.proj_v = nn.Linear(visual_dim, proj_dim)
        self.proj_a = nn.Linear(audio_dim, proj_dim)
        self.proj_t = nn.Linear(text_dim, proj_dim)

        # Discrepancy features: 3 scalar discrepancy scores + cross-attention interaction
        # Total concatenated input dim: visual_dim + audio_dim + text_dim + 3 (discrepancy scores) + proj_dim (fused cross context)
        in_features = visual_dim + audio_dim + text_dim + 3 + proj_dim

        # 3-Layer MLP with Residuals, Dropout (p=0.3), and ReLU
        self.fc1 = nn.Linear(in_features, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])

        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])

        # Residual shortcut from layer 1 to layer 2
        self.residual_proj = nn.Linear(hidden_dims[0], hidden_dims[1]) if hidden_dims[0] != hidden_dims[1] else nn.Identity()

        # Final classification logits (3 classes)
        self.classifier = nn.Linear(hidden_dims[2], num_classes)

    def compute_discrepancy(
        self,
        v_norm: torch.Tensor,
        a_norm: torch.Tensor,
        t_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the inter-modality discrepancy scores and cross-multiplication context.
        """
        # Project into common metric space
        pv = F.normalize(self.proj_v(v_norm), p=2, dim=-1)  # (B, proj_dim)
        pa = F.normalize(self.proj_a(a_norm), p=2, dim=-1)  # (B, proj_dim)
        pt = F.normalize(self.proj_t(t_norm), p=2, dim=-1)  # (B, proj_dim)

        # Dot-product similarities (in range [-1, 1])
        sim_va = torch.sum(pv * pa, dim=-1, keepdim=True)  # (B, 1)
        sim_vt = torch.sum(pv * pt, dim=-1, keepdim=True)  # (B, 1)
        sim_at = torch.sum(pa * pt, dim=-1, keepdim=True)  # (B, 1)

        # Discrepancy = 1.0 - similarity (0 = aligned, 2 = maximally contradicting)
        disc_va = 1.0 - sim_va
        disc_vt = 1.0 - sim_vt
        disc_at = 1.0 - sim_at
        discrepancy_scalars = torch.cat([disc_va, disc_vt, disc_at], dim=-1)  # (B, 3)

        # Cross-modality contextual interaction vector
        cross_context = (pv * pa) + (pv * pt) + (pa * pt)  # (B, proj_dim)

        metrics = {
            "disc_visual_audio": disc_va,
            "disc_visual_text": disc_vt,
            "disc_audio_text": disc_at
        }
        return discrepancy_scalars, cross_context, metrics

    def forward(
        self,
        v_f: torch.Tensor,
        a_f: torch.Tensor,
        t_f: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        Args:
            v_f: (B, visual_dim)
            a_f: (B, audio_dim)
            t_f: (B, text_dim)
        Returns:
            logits: (B, 3)
            discrepancy_dict: dictionary of discrepancy scores
        """
        # Layer Normalization
        v_norm = self.ln_v(v_f)
        a_norm = self.ln_a(a_f)
        t_norm = self.ln_t(t_f)

        # Compute discrepancy terms
        disc_scalars, cross_context, disc_dict = self.compute_discrepancy(v_norm, a_norm, t_norm)

        # Concatenate into unified multimodal tensor
        f_cat = torch.cat([v_norm, a_norm, t_norm, disc_scalars, cross_context], dim=-1)

        # MLP with residual connection & dropout
        h1 = self.dropout(self.relu(self.bn1(self.fc1(f_cat))))
        res1 = self.residual_proj(h1)
        h2 = self.dropout(self.relu(self.bn2(self.fc2(h1)) + res1))
        h3 = self.relu(self.bn3(self.fc3(h2)))

        logits = self.classifier(h3)
        return logits, disc_dict


class VeriMediaTeacherModel(nn.Module):
    """
    Full High-Capacity Teacher Ensemble.
    Uses ResNet50/EfficientNet visual backbone with SRM, Acoustic Vocoder backbone,
    and Intent Claim NLP backbone connected to Late Fusion MLP Head.
    """
    def __init__(self):
        super().__init__()
        self.visual_backbone = TeacherVisualBackbone(feature_dim=2048, use_srm=True)
        self.audio_backbone = AcousticVocoderBackbone(out_dim=768)
        self.nlp_backbone = IntentClaimBackbone(out_dim=768)
        
        self.fusion_head = MultimodalLateFusionHead(
            visual_dim=2048,
            audio_dim=768,
            text_dim=768,
            proj_dim=256,
            num_classes=3,
            dropout=0.3
        )

    def forward(
        self,
        frames: torch.Tensor,
        audio: torch.Tensor,
        text_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        v_f = self.visual_backbone(frames)
        a_f = self.audio_backbone(audio)
        t_f = self.nlp_backbone(text_tokens)
        
        logits, disc_dict = self.fusion_head(v_f, a_f, t_f)
        return logits, disc_dict


class VeriMediaStudentModel(nn.Module):
    """
    Distilled Edge Student Model.
    Uses MobileNetV3 with SRM, lightweight acoustic encoder, and edge claim net.
    Engineered for sub-300ms latency, <=45MB ONNX quantized binary size.
    """
    def __init__(self, visual_dim: int = 576, audio_dim: int = 768, text_dim: int = 768):
        super().__init__()
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        
        self.visual_backbone = StudentVisualBackbone(out_dim=visual_dim, use_srm=True)
        self.audio_backbone = AcousticVocoderBackbone(out_dim=audio_dim)
        self.nlp_backbone = IntentClaimBackbone(out_dim=text_dim)
        
        self.fusion_head = MultimodalLateFusionHead(
            visual_dim=visual_dim,
            audio_dim=audio_dim,
            text_dim=text_dim,
            proj_dim=128,
            hidden_dims=[256, 128, 64],
            num_classes=3,
            dropout=0.2
        )

    def forward(
        self,
        frames: torch.Tensor,
        audio: torch.Tensor,
        text_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        v_f = self.visual_backbone(frames)
        a_f = self.audio_backbone(audio)
        t_f = self.nlp_backbone(text_tokens)
        
        logits, disc_dict = self.fusion_head(v_f, a_f, t_f)
        return logits, disc_dict


class VeriMediaEdgeONNXWrapper(nn.Module):
    """
    Simplified flattened wrapper for ONNX export and INT8 quantization.
    Takes pre-extracted / pooled feature tensors (V_f, A_f, T_f) or single frame + audio + tokens.
    """
    def __init__(self, fusion_head: MultimodalLateFusionHead):
        super().__init__()
        self.fusion_head = fusion_head

    def forward(self, v_f: torch.Tensor, a_f: torch.Tensor, t_f: torch.Tensor) -> torch.Tensor:
        logits, _ = self.fusion_head(v_f, a_f, t_f)
        probs = F.softmax(logits, dim=-1)
        return probs
