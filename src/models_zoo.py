"""
models_zoo.py - VeriMedia Model Zoo & Alternative Deepfake Architectures
Implements XceptionNet, EfficientNetV2, ResNet-50, MesoInception-4, and Temporal-LSTM.
All models integrate Steganalysis Rich Model (SRM) frequency filter front-ends.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Dict, Any, Optional, Tuple

from src.srm_filter import SRMConv2d
from src.audio_branch import AcousticVocoderBackbone
from src.nlp_branch import IntentClaimBackbone
from src.fusion_model import MultimodalLateFusionHead


# =====================================================================
# 1. XceptionNet Backbone (Official FaceForensics++ Benchmark)
# =====================================================================
class XceptionBackbone(nn.Module):
    """
    XceptionNet forensic backbone with Steganalysis Rich Model (SRM) front-end.
    Canonical baseline used in FaceForensics++ benchmark research.
    """
    def __init__(self, out_dim: int = 1024, use_srm: bool = True, pretrained: bool = False):
        super().__init__()
        self.out_dim = out_dim
        self.use_srm = use_srm
        self.srm = SRMConv2d(in_channels=3, out_mode="rgb_residual") if use_srm else nn.Identity()

        # Load Xception backbone via timm
        try:
            self.base = timm.create_model("legacy_xception", pretrained=pretrained, num_classes=0)
        except Exception:
            try:
                self.base = timm.create_model("xception", pretrained=False, num_classes=0)
            except Exception:
                self.base = timm.create_model("legacy_xception", pretrained=False, num_classes=0)
        num_features = self.base.num_features  # 2048

        self.proj = nn.Linear(num_features, out_dim) if out_dim != num_features else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            b, n, c, h, w = x.shape
            x = x.view(b * n, c, h, w)
            has_frames = True
        else:
            b = x.shape[0]
            has_frames = False

        if self.use_srm:
            residuals = self.srm(x)
            x = x + 0.3 * residuals

        feat = self.base(x)  # (B*N, num_features)
        feat = self.proj(feat)

        if has_frames:
            feat = feat.view(b, -1, self.out_dim)
            feat = torch.mean(feat, dim=1)
        return feat


# =====================================================================
# 2. EfficientNetV2 Backbone (Kaggle DFDC Top Performer)
# =====================================================================
class EfficientNetV2Backbone(nn.Module):
    """
    EfficientNetV2-S with progressive learning and SRM frequency extraction.
    """
    def __init__(self, out_dim: int = 1280, use_srm: bool = True, pretrained: bool = False):
        super().__init__()
        self.out_dim = out_dim
        self.use_srm = use_srm
        self.srm = SRMConv2d(in_channels=3, out_mode="rgb_residual") if use_srm else nn.Identity()

        weights = models.EfficientNet_V2_S_Weights.DEFAULT if (pretrained and hasattr(models, "EfficientNet_V2_S_Weights")) else None
        base = models.efficientnet_v2_s(weights=weights)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        in_features = 1280
        self.proj = nn.Linear(in_features, out_dim) if out_dim != in_features else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            b, n, c, h, w = x.shape
            x = x.view(b * n, c, h, w)
            has_frames = True
        else:
            b = x.shape[0]
            has_frames = False

        if self.use_srm:
            residuals = self.srm(x)
            x = x + 0.3 * residuals

        out = self.features(x)
        out = self.avgpool(out)
        feat = torch.flatten(out, 1)
        feat = self.proj(feat)

        if has_frames:
            feat = feat.view(b, -1, self.out_dim)
            feat = torch.mean(feat, dim=1)
        return feat


# =====================================================================
# 3. ResNet-50 Backbone (Deep Residual Forensic Network)
# =====================================================================
class ResNet50Backbone(nn.Module):
    """
    ResNet-50 with residual connections and SRM frequency front-end.
    """
    def __init__(self, out_dim: int = 1024, use_srm: bool = True, pretrained: bool = False):
        super().__init__()
        self.out_dim = out_dim
        self.use_srm = use_srm
        self.srm = SRMConv2d(in_channels=3, out_mode="rgb_residual") if use_srm else nn.Identity()

        weights = models.ResNet50_Weights.DEFAULT if (pretrained and hasattr(models, "ResNet50_Weights")) else None
        base = models.resnet50(weights=weights)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.proj = nn.Linear(2048, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            b, n, c, h, w = x.shape
            x = x.view(b * n, c, h, w)
            has_frames = True
        else:
            b = x.shape[0]
            has_frames = False

        if self.use_srm:
            residuals = self.srm(x)
            x = x + 0.3 * residuals

        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        feat = torch.flatten(out, 1)
        feat = self.proj(feat)

        if has_frames:
            feat = feat.view(b, -1, self.out_dim)
            feat = torch.mean(feat, dim=1)
        return feat


# =====================================================================
# 4. MesoInception-4 Backbone (Classic Microscopic Forensic Model)
# =====================================================================
class InceptionBlock(nn.Module):
    def __init__(self, in_c: int, a: int, b: int, c: int, d: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, a, kernel_size=1, padding=0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, b, kernel_size=1, padding=0),
            nn.Conv2d(b, b, kernel_size=3, padding=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_c, c, kernel_size=1, padding=0),
            nn.Conv2d(c, c, kernel_size=3, padding=2, dilation=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_c, d, kernel_size=1, padding=0),
            nn.Conv2d(d, d, kernel_size=3, padding=3, dilation=3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)


class MesoInception4Backbone(nn.Module):
    """
    MesoInception-4: Specially designed to detect microscopic spatial inconsistencies
    and mesoscopic compression noise patterns.
    """
    def __init__(self, out_dim: int = 512, use_srm: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.use_srm = use_srm
        self.srm = SRMConv2d(in_channels=3, out_mode="rgb_residual") if use_srm else nn.Identity()

        self.inc1 = InceptionBlock(3, 1, 4, 4, 2)    # 11 channels
        self.bn1 = nn.BatchNorm2d(11)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.inc2 = InceptionBlock(11, 2, 4, 4, 2)   # 12 channels
        self.bn2 = nn.BatchNorm2d(12)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(12, 16, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(4, 4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            b, n, c, h, w = x.shape
            x = x.view(b * n, c, h, w)
            has_frames = True
        else:
            b = x.shape[0]
            has_frames = False

        if self.use_srm:
            residuals = self.srm(x)
            x = x + 0.3 * residuals

        out = self.pool1(nn.functional.relu(self.bn1(self.inc1(x))))
        out = self.pool2(nn.functional.relu(self.bn2(self.inc2(out))))
        out = self.pool3(nn.functional.relu(self.bn3(self.conv3(out))))
        out = self.pool4(nn.functional.relu(self.bn4(self.conv4(out))))
        out = self.avgpool(out)
        feat = torch.flatten(out, 1)
        feat = self.fc(feat)

        if has_frames:
            feat = feat.view(b, -1, self.out_dim)
            feat = torch.mean(feat, dim=1)
        return feat


# =====================================================================
# 5. Temporal-LSTM Backbone (Multi-Frame Video Flickering Detector)
# =====================================================================
class TemporalLSTMBackbone(nn.Module):
    """
    Spatial-Temporal Backbone: Passes frames through a 2D CNN,
    then evaluates frame-to-frame temporal dynamics using a Bidirectional LSTM.
    Catches unnatural blinking, facial jitter, and boundary flickering over time.
    """
    def __init__(self, out_dim: int = 512, use_srm: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.use_srm = use_srm
        self.srm = SRMConv2d(in_channels=3, out_mode="rgb_residual") if use_srm else nn.Identity()

        # 2D CNN frame feature extractor
        base = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if hasattr(models, 'MobileNet_V3_Small_Weights') else None
        )
        self.cnn = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Temporal Bidirectional LSTM
        self.lstm = nn.LSTM(input_size=576, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            # Add temporal dimension N=1
            x = x.unsqueeze(1)

        b, n, c, h, w = x.shape
        x_flat = x.view(b * n, c, h, w)

        if self.use_srm:
            residuals = self.srm(x_flat)
            x_flat = x_flat + 0.3 * residuals

        spatial_feat = self.pool(self.cnn(x_flat))
        spatial_feat = torch.flatten(spatial_feat, 1)  # (B*N, 576)
        
        # Reshape to sequence: (B, N, 576)
        seq_feat = spatial_feat.view(b, n, 576)
        lstm_out, _ = self.lstm(seq_feat)  # (B, N, 512)

        # Temporal pooling over frames
        temp_pooled = torch.mean(lstm_out, dim=1)  # (B, 512)
        feat = self.fc(temp_pooled)  # (B, out_dim)
        return feat


# =====================================================================
# Complete End-to-End Multimodal Model Wrapper for Any Backbone
# =====================================================================
class VeriMediaZooModel(nn.Module):
    """
    Unified Multimodal Model connecting any Visual Backbone to Audio, NLP,
    and the Late-Fusion Discrepancy Head.
    """
    def __init__(self, backbone_type: str = "mobilenet_v3", visual_dim: Optional[int] = None, pretrained: bool = False):
        super().__init__()
        self.backbone_type = backbone_type.lower()

        # Instantiate visual backbone
        if self.backbone_type in ["xception", "xceptionnet"]:
            dim = visual_dim or 1024
            self.visual_backbone = XceptionBackbone(out_dim=dim, use_srm=True, pretrained=pretrained)
        elif self.backbone_type in ["efficientnet_v2", "efficientnet"]:
            dim = visual_dim or 1280
            self.visual_backbone = EfficientNetV2Backbone(out_dim=dim, use_srm=True, pretrained=pretrained)
        elif self.backbone_type in ["resnet50", "resnet"]:
            dim = visual_dim or 1024
            self.visual_backbone = ResNet50Backbone(out_dim=dim, use_srm=True, pretrained=pretrained)
        elif self.backbone_type in ["mesonet", "mesoinception"]:
            dim = visual_dim or 512
            self.visual_backbone = MesoInception4Backbone(out_dim=dim, use_srm=True)
        elif self.backbone_type in ["temporal_lstm", "lstm", "temporal"]:
            dim = visual_dim or 512
            self.visual_backbone = TemporalLSTMBackbone(out_dim=dim, use_srm=True)
        else:
            # Default edge MobileNetV3
            from src.visual_branch import StudentVisualBackbone
            dim = visual_dim or 576
            self.visual_backbone = StudentVisualBackbone(out_dim=dim, use_srm=True)

        self.visual_dim = dim
        self.audio_dim = 768
        self.text_dim = 768

        self.audio_backbone = AcousticVocoderBackbone(out_dim=self.audio_dim)
        self.nlp_backbone = IntentClaimBackbone(out_dim=self.text_dim)

        self.fusion_head = MultimodalLateFusionHead(
            visual_dim=self.visual_dim,
            audio_dim=self.audio_dim,
            text_dim=self.text_dim,
            proj_dim=128,
            hidden_dims=[256, 128, 64],
            num_classes=3,
            dropout=0.2
        )

    def forward(self, frames: torch.Tensor, audio: torch.Tensor, tokens: torch.Tensor):
        v_f = self.visual_backbone(frames)
        a_f = self.audio_backbone(audio)
        t_f = self.nlp_backbone(tokens)
        logits, disc_dict = self.fusion_head(v_f, a_f, t_f)
        return logits, disc_dict
