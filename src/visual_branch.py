"""
visual_branch.py - Visual Spatial & Frequency Forensic Backbone
Processes video streams at 5 FPS, applies SRM high-pass filtering,
and extracts 2048d (Teacher) and edge-optimized (Student) spatial-frequency embeddings.
"""

import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from src.srm_filter import SRMConv2d
from src.config import config


class TeacherVisualBackbone(nn.Module):
    """
    Teacher Visual Backbone (EfficientNetV2 / ResNet50 + SRM).
    Produces a 2048-dimensional spatial-frequency feature vector (V_f).
    """
    def __init__(self, feature_dim: int = 2048, use_srm: bool = True):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_srm = use_srm
        
        # SRM high-pass filter layer
        self.srm = SRMConv2d(in_channels=3, out_mode="rgb_residual") if use_srm else nn.Identity()

        # ResNet50 / EfficientNet backbone
        # Using ResNet50 as standard high-capacity backbone with exactly 2048d final pooling
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if hasattr(models, 'ResNet50_Weights') else None)
        
        # Extract convolutional layers prior to classification fc
        self.conv_stem = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        )
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Linear projection if target feature_dim != 2048
        if feature_dim != 2048:
            self.proj = nn.Linear(2048, feature_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, 3, H, W) or (B, N, 3, H, W)
        Returns:
            V_f: Tensor of shape (B, feature_dim)
        """
        # If input has frame dimension (B, N, C, H, W)
        if x.ndim == 5:
            b, n, c, h, w = x.shape
            x = x.view(b * n, c, h, w)
            has_frames = True
        else:
            b = x.shape[0]
            has_frames = False

        # Apply SRM frequency filter
        if self.use_srm:
            # SRM extracts high-frequency residuals
            residuals = self.srm(x)
            # Combine RGB with high-pass residuals
            x_in = x + 0.3 * residuals
        else:
            x_in = x

        out = self.conv_stem(x_in)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        features = torch.flatten(out, 1)  # (B*N, 2048)
        features = self.proj(features)

        if has_frames:
            # Pool across sampled frames (B, N, feature_dim) -> (B, feature_dim)
            features = features.view(b, -1, self.feature_dim)
            features = torch.mean(features, dim=1)

        return features


class StudentVisualBackbone(nn.Module):
    """
    Student Visual Backbone (MobileNetV3 + SRM).
    Lightweight, edge-compliant spatial-frequency feature extractor.
    """
    def __init__(self, out_dim: int = 576, use_srm: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.use_srm = use_srm
        
        self.srm = SRMConv2d(in_channels=3, out_mode="rgb_residual") if use_srm else nn.Identity()
        
        # MobileNetV3-Small backbone
        base_mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if hasattr(models, 'MobileNet_V3_Small_Weights') else None
        )
        self.features = base_mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MobileNetV3 small outputs 576-dim feature from features
        in_features = 576
        if out_dim != in_features:
            self.proj = nn.Linear(in_features, out_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for student visual branch.
        """
        if x.ndim == 5:
            b, n, c, h, w = x.shape
            x = x.view(b * n, c, h, w)
            has_frames = True
        else:
            b = x.shape[0]
            has_frames = False

        if self.use_srm:
            residuals = self.srm(x)
            x_in = x + 0.3 * residuals
        else:
            x_in = x

        feat = self.features(x_in)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        feat = self.proj(feat)

        if has_frames:
            feat = feat.view(b, -1, self.out_dim)
            feat = torch.mean(feat, dim=1)

        return feat


class VideoFrameExtractor:
    """
    Extracts frames from video files and crops detected facial regions using
    a multi-cascade Haar ensemble (frontal alt2/default/alt + left/right profile)
    with histogram equalization and bidirectional bounding-box propagation.
    """
    def __init__(self, target_fps: int = 5, max_frames: int = 32, crop_size: int = 224):
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.crop_size = crop_size
        self.last_extraction_info: Dict[str, Any] = {
            "faces_detected_frames": 0,
            "total_sampled_frames": 0,
            "face_detected": False,
        }

        # Load OpenCV multi-angle Haar cascade face detectors with graceful fallback
        self.face_cascade = None
        self._cascades: List[Tuple[str, Any]] = []
        self._profile_cascade = None
        try:
            if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades') and hasattr(cv2, 'CascadeClassifier'):
                cascade_files = [
                    ("frontal_default", "haarcascade_frontalface_default.xml"),
                    ("frontal_alt2", "haarcascade_frontalface_alt2.xml"),
                    ("frontal_alt", "haarcascade_frontalface_alt.xml"),
                    ("profile", "haarcascade_profileface.xml"),
                ]
                for cname, fname in cascade_files:
                    cpath = cv2.data.haarcascades + fname
                    if os.path.exists(cpath):
                        clf = cv2.CascadeClassifier(cpath)
                        if not clf.empty():
                            self._cascades.append((cname, clf))
                            if cname == "frontal_default" or self.face_cascade is None:
                                self.face_cascade = clf
                            if cname == "profile":
                                self._profile_cascade = clf
        except Exception:
            self.face_cascade = None
            self._cascades = []

        # Standard ImageNet normalization transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def _skin_ratio(bgr_crop: np.ndarray) -> float:
        """Fast YCrCb skin-tone ratio check to reject Haar false-positives on pavement/foliage/text."""
        if bgr_crop is None or bgr_crop.size == 0:
            return 0.0
        ycrcb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(
            ycrcb,
            np.array([25, 133, 77], dtype=np.uint8),
            np.array([255, 175, 130], dtype=np.uint8),
        )
        return float((mask > 0).mean())

    @staticmethod
    def _detect_with_weights(
        clf: Any,
        gray_img: np.ndarray,
        scaleFactor: float,
        minNeighbors: int,
        minSize: Tuple[int, int],
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Runs detectMultiScale3 when available to expose confidence weights alongside rects."""
        try:
            rects, _, weights = clf.detectMultiScale3(
                gray_img,
                scaleFactor=scaleFactor,
                minNeighbors=minNeighbors,
                minSize=minSize,
                outputRejectLevels=True,
            )
            if len(rects) == 0:
                return []
            wt_list = [float(w) for w in weights] if len(weights) == len(rects) else [2.0] * len(rects)
            return [(tuple(int(v) for v in r), w) for r, w in zip(rects, wt_list)]
        except Exception:
            try:
                rects = clf.detectMultiScale(
                    gray_img, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize
                )
                return [(tuple(int(v) for v in r), 2.0) for r in rects]
            except Exception:
                return []

    def _pick_valid_box(
        self,
        frame: np.ndarray,
        candidates: List[Tuple[Tuple[int, int, int, int], float]],
        scale: float,
        min_weight: float,
        min_skin: float,
        flip_x: bool = False,
        scaled_w: int = 0,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Selects the largest detected box passing size, Haar weight, and skin-ratio checks."""
        if not candidates:
            return None
        h, w = frame.shape[:2]
        min_px = max(24, min(110, int(0.05 * min(h, w))))
        ordered = sorted(candidates, key=lambda item: item[0][2] * item[0][3], reverse=True)
        for (fx_s, fy_s, fw_s, fh_s), wt in ordered:
            if wt < min_weight:
                continue
            if flip_x and scaled_w > 0:
                fx_s = scaled_w - fx_s - fw_s
            fx, fy, fw, fh = [int(v / scale) for v in (fx_s, fy_s, fw_s, fh_s)]
            if max(fw, fh) < min_px:
                continue
            x1, y1, x2, y2 = max(0, fx), max(0, fy), min(w, fx + fw), min(h, fy + fh)
            sk = self._skin_ratio(frame[y1:y2, x1:x2])
            if sk < min_skin and not (wt >= max(min_weight * 2.5, 5.0) and sk >= 0.14):
                continue
            mx, my = int(0.2 * fw), int(0.2 * fh)
            return (max(0, fx - mx), max(0, fy - my), min(w, fx + fw + mx), min(h, fy + fh + my))
        return None

    def _detect_best_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Tries training-aligned frontal_default@320px first, then skin-validated equalized cascades."""
        if not self._cascades:
            return None
        h, w = frame.shape[:2]
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) Exact match with frame_cache.py so trained FF++ face crop geometry is preserved 1:1
        if self.face_cascade is not None:
            sc_tr = 320.0 / max(w, 1)
            sh_tr = max(1, int(round(h * sc_tr)))
            small_tr = cv2.resize(gray_full, (320, sh_tr))
            cands = self._detect_with_weights(
                self.face_cascade, small_tr, scaleFactor=1.15, minNeighbors=4, minSize=(20, 20)
            )
            box = self._pick_valid_box(frame, cands, sc_tr, min_weight=1.2, min_skin=0.22)
            if box is not None:
                return box

        # 2) Multi-angle histogram-equalized fallback for portrait/low-light/profile uploads
        scale = 480.0 / max(w, h, 1)
        sw = max(1, int(round(w * scale)))
        sh = max(1, int(round(h * scale)))
        eq = cv2.equalizeHist(cv2.resize(gray_full, (sw, sh)))

        for cname, clf in self._cascades:
            cands = self._detect_with_weights(
                clf, eq, scaleFactor=1.08, minNeighbors=3, minSize=(24, 24)
            )
            min_w = 2.5 if cname == "frontal_alt2" else 1.5
            min_s = 0.28 if cname == "profile" else 0.25
            box = self._pick_valid_box(frame, cands, scale, min_weight=min_w, min_skin=min_s)
            if box is not None:
                return box

        if self._profile_cascade is not None:
            eq_flip = cv2.flip(eq, 1)
            cands = self._detect_with_weights(
                self._profile_cascade, eq_flip, scaleFactor=1.08, minNeighbors=3, minSize=(24, 24)
            )
            box = self._pick_valid_box(
                frame, cands, scale, min_weight=1.5, min_skin=0.28, flip_x=True, scaled_w=sw
            )
            if box is not None:
                return box

        return None

    def extract_and_preprocess(self, video_path: str) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """
        Extracts frames at target_fps, detects/crops skin-validated faces with outlier filtering,
        and returns:
            frames_tensor: (1, N, 3, crop_size, crop_size)
            raw_crops: list of unnormalized RGB uint8 images for visualization
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.last_extraction_info = {
                "faces_detected_frames": 0,
                "total_sampled_frames": 0,
                "face_detected": False,
                "sufficient_face_evidence": False,
            }
            dummy = torch.zeros((1, 1, 3, self.crop_size, self.crop_size), dtype=torch.float32)
            return dummy, [np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)]

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or np.isnan(video_fps):
            video_fps = 25.0

        span_sec = max(1.6, float(self.max_frames) / max(1.0, float(self.target_fps)))
        if total_frames <= 0:
            span_limit = max(self.max_frames * 4, int(round(video_fps * span_sec)))
        else:
            span_limit = min(total_frames, max(self.max_frames, int(round(video_fps * span_sec))))

        targets = set(np.linspace(0, max(0, span_limit - 1), self.max_frames, dtype=int).tolist())
        max_idx = max(targets) if targets else 0

        sampled_frames: List[np.ndarray] = []
        detected_boxes: List[Optional[Tuple[int, int, int, int]]] = []
        for frame_idx in range(max_idx + 1):
            if not cap.grab():
                break
            if frame_idx not in targets:
                continue
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                continue
            sampled_frames.append(frame)
            detected_boxes.append(self._detect_best_face(frame))
        cap.release()

        if not sampled_frames:
            self.last_extraction_info = {
                "faces_detected_frames": 0,
                "total_sampled_frames": 0,
                "face_detected": False,
                "sufficient_face_evidence": False,
            }
            dummy = torch.zeros((1, 1, 3, self.crop_size, self.crop_size), dtype=torch.float32)
            return dummy, [np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)]

        # Reject isolated shot-cut / scale-jump outliers when >= 3 valid boxes exist
        valid_items = [(i, b) for i, b in enumerate(detected_boxes) if b is not None]
        if len(valid_items) >= 3:
            cxs = np.array([(b[0] + b[2]) * 0.5 for _, b in valid_items])
            cys = np.array([(b[1] + b[3]) * 0.5 for _, b in valid_items])
            areas = np.array([max(1, (b[2] - b[0]) * (b[3] - b[1])) for _, b in valid_items])
            med_cx, med_cy, med_area = float(np.median(cxs)), float(np.median(cys)), float(np.median(areas))
            fh0, fw0 = sampled_frames[0].shape[:2]
            max_jump = 0.45 * max(fh0, fw0)
            for idx_b, b in valid_items:
                cx = (b[0] + b[2]) * 0.5
                cy = (b[1] + b[3]) * 0.5
                area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
                if np.hypot(cx - med_cx, cy - med_cy) > max_jump or area < 0.45 * med_area or area > 2.0 * med_area:
                    detected_boxes[idx_b] = None

        hits = sum(1 for b in detected_boxes if b is not None)
        n_sampled = len(sampled_frames)
        min_required = 1 if n_sampled <= 2 else max(2, min(3, (n_sampled + 1) // 2))
        self.last_extraction_info = {
            "faces_detected_frames": hits,
            "total_sampled_frames": n_sampled,
            "face_detected": hits > 0,
            "sufficient_face_evidence": hits >= min_required,
        }

        # Build verified crops first; for undetected frames only accept propagated box if skin-validated,
        # otherwise carry nearest verified face crop so scene-cut backgrounds never poison temporal pooling
        filled_boxes = list(detected_boxes)
        last_box = None
        for i, b in enumerate(filled_boxes):
            if b is not None:
                last_box = b
            elif last_box is not None:
                filled_boxes[i] = last_box
        next_box = None
        for i in range(len(filled_boxes) - 1, -1, -1):
            if filled_boxes[i] is not None:
                next_box = filled_boxes[i]
            elif next_box is not None:
                filled_boxes[i] = next_box

        verified_crops: List[Optional[np.ndarray]] = [None] * n_sampled
        ref_hists: List[np.ndarray] = []
        for i, (frame, raw_b) in enumerate(zip(sampled_frames, detected_boxes)):
            if raw_b is not None:
                x1, y1, x2, y2 = raw_b
                vc = frame[y1:y2, x1:x2]
                verified_crops[i] = vc
                if vc.size > 0:
                    hsv = cv2.cvtColor(cv2.resize(vc, (64, 64)), cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                    ref_hists.append(hist)

        for i, (frame, raw_b, fill_b) in enumerate(zip(sampled_frames, detected_boxes, filled_boxes)):
            if raw_b is None and fill_b is not None:
                x1, y1, x2, y2 = fill_b
                cand_crop = frame[y1:y2, x1:x2]
                if self._skin_ratio(cand_crop) >= 0.22 and cand_crop.size > 0:
                    if ref_hists:
                        hsv_c = cv2.cvtColor(cv2.resize(cand_crop, (64, 64)), cv2.COLOR_BGR2HSV)
                        hist_c = cv2.calcHist([hsv_c], [0, 1], None, [16, 16], [0, 180, 0, 256])
                        cv2.normalize(hist_c, hist_c, 0, 1, cv2.NORM_MINMAX)
                        best_sim = max(cv2.compareHist(h_r, hist_c, cv2.HISTCMP_CORREL) for h_r in ref_hists)
                        if best_sim >= 0.50:
                            verified_crops[i] = cand_crop
                    else:
                        verified_crops[i] = cand_crop

        last_crop = None
        for i in range(n_sampled):
            if verified_crops[i] is not None:
                last_crop = verified_crops[i]
            elif last_crop is not None:
                verified_crops[i] = last_crop
        next_crop = None
        for i in range(n_sampled - 1, -1, -1):
            if verified_crops[i] is not None:
                next_crop = verified_crops[i]
            elif next_crop is not None:
                verified_crops[i] = next_crop

        frames: List[torch.Tensor] = []
        raw_crops: List[np.ndarray] = []
        mean_arr = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std_arr = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for frame, v_crop in zip(sampled_frames, verified_crops):
            h, w = frame.shape[:2]
            if v_crop is not None and v_crop.size > 0:
                crop = v_crop
            else:
                min_dim = min(h, w)
                cy, cx = h // 2, w // 2
                crop = frame[cy - min_dim // 2 : cy + min_dim // 2, cx - min_dim // 2 : cx + min_dim // 2]

            if crop.size == 0:
                crop = frame

            crop_rgb = cv2.cvtColor(cv2.resize(crop, (self.crop_size, self.crop_size)), cv2.COLOR_BGR2RGB)
            raw_crops.append(crop_rgb)
            arr_f = (crop_rgb.astype(np.float32) / 255.0 - mean_arr) / std_arr
            tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(arr_f, (2, 0, 1))))
            frames.append(tensor)

        if not frames:
            dummy = torch.zeros((1, 1, 3, self.crop_size, self.crop_size), dtype=torch.float32)
            return dummy, [np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)]

        stacked = torch.stack(frames, dim=0).unsqueeze(0)
        return stacked, raw_crops
