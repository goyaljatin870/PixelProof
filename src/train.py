"""
train.py - Multimodal Late-Fusion & Knowledge Distillation Training Script
Trains Late-Fusion MLP classification head and distills Teacher Ensemble into Edge Student Model
using FaceForensics++ data manifest, cached face crops, balanced sampling, and CUDA AMP.
"""

import os
import sys
import re
import zlib
import json
import argparse
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Ensure package root is in sys.path
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.config import config
from src.visual_branch import VideoFrameExtractor, StudentVisualBackbone
from src.audio_branch import AudioBranchExtractor, AcousticVocoderBackbone
from src.nlp_branch import IntentClaimBackbone, MISINFORMATION_LEXICON
from src.fusion_model import VeriMediaTeacherModel, VeriMediaStudentModel, MultimodalLateFusionHead
from src.distillation import DistillationLoss
from src.quantization import export_student_to_onnx, quantize_onnx_to_int8

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Regional misinformation claims for contextual deception simulation
DECEPTIVE_SAMPLE_CLAIMS = [
    "LEAKED: Confidential meeting recording exposed before regional election! Share now!!",
    "URGENT: Government officially confirms secret currency withdrawal, viral video!",
    "Caught on camera! Opposition candidate secretly accepting cash, watch before deleted!",
    "Exposed: Deepfake audio proves election voting machines were hacked!",
    "BREAKING: Hidden truth revealed in leaked insider speech, forward before censored!"
]

BENIGN_SAMPLE_CLAIMS = [
    "Official parliamentary discussion on annual economic infrastructure development.",
    "Press conference briefing regarding local agricultural policy updates.",
    "Public address by municipal authorities regarding civic improvement projects.",
    "Educational documentary clip on regional cultural heritage.",
    "Recorded interview statement shared during regional civic forum."
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_GPU_STATS_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}


def gpu_prepare_frames(
    frames: torch.Tensor,
    device: str,
    augment: bool = False,
    domain_augment: bool = False,
) -> torch.Tensor:
    """
    Moves frames to target device and converts cached uint8 (B, N, H, W, 3) tensors
    to normalized float32 (B, N, 3, H, W) directly on GPU (~4x less CPU & PCIe traffic).
    Also supports legacy float32 (B, N, 3, H, W) inputs transparently.
    """
    dev_key = str(device)
    t = frames.to(device, non_blocking=True)
    if t.dtype != torch.uint8:
        return t

    t = t.permute(0, 1, 4, 2, 3).contiguous().float().mul_(1.0 / 255.0)
    if augment:
        b = t.shape[0]
        if domain_augment:
            contrast = torch.empty((b, 1, 1, 1, 1), device=t.device, dtype=t.dtype).uniform_(0.90, 1.12)
            shift = torch.empty((b, 1, 1, 1, 1), device=t.device, dtype=t.dtype).uniform_(-0.04, 0.04)
            tint = torch.empty((b, 1, 3, 1, 1), device=t.device, dtype=t.dtype).uniform_(0.96, 1.04)
            sigma = torch.empty((b, 1, 1, 1, 1), device=t.device, dtype=t.dtype).uniform_(0.0, 0.012)
            noise = torch.randn_like(t) * sigma
            t.sub_(0.5).mul_(contrast).add_(0.5).add_(shift).mul_(tint).add_(noise).clamp_(0.0, 1.0)
        else:
            gain = torch.empty((b, 1, 1, 1, 1), device=t.device, dtype=t.dtype).uniform_(0.90, 1.10)
            t.mul_(gain).clamp_(0.0, 1.0)

    if dev_key not in _GPU_STATS_CACHE:
        mean_t = torch.tensor([0.485, 0.456, 0.406], device=t.device, dtype=torch.float32).view(1, 1, 3, 1, 1)
        std_t = torch.tensor([0.229, 0.224, 0.225], device=t.device, dtype=torch.float32).view(1, 1, 3, 1, 1)
        _GPU_STATS_CACHE[dev_key] = (mean_t, std_t)
    mean_t, std_t = _GPU_STATS_CACHE[dev_key]
    t.sub_(mean_t).div_(std_t)
    return t


def _apply_clip_domain_degradation(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Applies realistic clip-consistent compression, rescale blur, and crop jitter
    so backbones stop overfitting to FaceForensics++ C23 H.264 codec artifacts.
    """
    import cv2
    if arr.shape[0] == 0:
        return arr

    out = arr.copy()
    t_len, h, w, _ = out.shape

    # 1) Mild consistent crop zoom jitter across all frames
    if rng.random() < 0.45:
        scale_c = rng.uniform(0.86, 0.97)
        ch, cw = max(32, int(h * scale_c)), max(32, int(w * scale_c))
        y0 = rng.randint(0, max(0, h - ch))
        x0 = rng.randint(0, max(0, w - cw))
        for i in range(t_len):
            out[i] = cv2.resize(out[i, y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)

    # 2) Social-media resolution downscale -> upscale (covers soft 240p-480p mobile clips)
    if rng.random() < 0.70:
        side = rng.randint(96, 184)
        interp = cv2.INTER_AREA if rng.random() < 0.5 else cv2.INTER_LINEAR
        for i in range(t_len):
            sm = cv2.resize(out[i], (side, side), interpolation=interp)
            out[i] = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)

    # 3) Mild H.264-like blur OR gentle unsharp filter
    op_roll = rng.random()
    if op_roll < 0.30:
        sigma = rng.uniform(0.6, 1.25)
        ksize = 3 if sigma < 1.0 else 5
        for i in range(t_len):
            out[i] = cv2.GaussianBlur(out[i], (ksize, ksize), sigma)
    elif op_roll < 0.45:
        kernel = np.array([[0, -0.25, 0], [-0.25, 2.0, -0.25], [0, -0.25, 0]], dtype=np.float32)
        for i in range(t_len):
            out[i] = cv2.filter2D(out[i], -1, kernel)

    # 4) YCbCr 4:2:0 chroma subsampling (matches WhatsApp/X H.264 color macroblocking)
    if rng.random() < 0.55:
        for i in range(t_len):
            ycc = cv2.cvtColor(out[i], cv2.COLOR_RGB2YCrCb)
            ch = cv2.resize(ycc[:, :, 1:], (max(16, w // 2), max(16, h // 2)), interpolation=cv2.INTER_AREA)
            ycc[:, :, 1:] = cv2.resize(ch, (w, h), interpolation=cv2.INTER_LINEAR)
            out[i] = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)

    # 5) Realistic DCT / macroblock compression simulation
    if rng.random() < 0.85:
        q = rng.randint(32, 86)
        enc_params = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        for i in range(t_len):
            bgr = cv2.cvtColor(out[i], cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, enc_params)
            if ok:
                dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if dec is not None:
                    out[i] = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

    return out


def load_frame_cache_index(cache_dir: Optional[Path] = None) -> Dict[str, str]:
    """Loads cached video_path -> .npy crop tensor mapping if available."""
    cdir = Path(cache_dir) if cache_dir else config.project_root / "data" / "frame_cache_8f"
    idx_file = cdir / "cache_index.json"
    if idx_file.exists():
        try:
            with open(idx_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning(f"Failed loading cache_index.json: {e}")
    return {}


def tokenize_claim(text: str, max_len: int = 128) -> torch.Tensor:
    """Stateless deterministic CRC32 tokenizer matching IntentClaimBackbone.simple_tokenize."""
    words = re.findall(r"\w+", text.lower()) if text else []
    tokens = [1]
    for w in words:
        tok = (zlib.crc32(w.encode("utf-8")) % 9998) + 2
        tokens.append(tok)
        if len(tokens) >= max_len - 1:
            break
    tokens.append(2)
    pad_len = max_len - len(tokens)
    if pad_len > 0:
        tokens.extend([0] * pad_len)
    return torch.tensor(tokens, dtype=torch.long)


class FaceForensicsMultimodalDataset(Dataset):
    """
    Multimodal Dataset for FaceForensics++ videos paired with balanced claim conditioning.
    Supports pre-extracted lossless uint8 .npy face crops with safe OpenCV fallback.
    """
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        max_samples: Optional[int] = None,
        seed: int = 42,
        cache_index: Optional[Dict[str, str]] = None,
        num_frames: int = 8,
        augment: bool = False,
        deterministic: bool = False,
        contextual_misinfo_prob: float = 0.18,
        force_claim_mode: Optional[str] = None,
        domain_augment: bool = False,
        domain_prob: float = 0.75,
    ):
        self.df = manifest_df.reset_index(drop=True)
        if max_samples is not None and len(self.df) > max_samples:
            self.df = self.df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

        self.cache_index = cache_index if cache_index is not None else load_frame_cache_index()
        self.num_frames = int(num_frames)
        self.augment = bool(augment)
        self.deterministic = bool(deterministic)
        self.contextual_misinfo_prob = float(contextual_misinfo_prob)
        self.force_claim_mode = force_claim_mode
        self.domain_augment = bool(domain_augment)
        self.domain_prob = float(domain_prob)

        self.frame_extractor = VideoFrameExtractor(
            target_fps=config.model.sample_fps,
            max_frames=min(config.model.max_frames_per_video, max(4, self.num_frames))
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_frames_tensor(self, v_path: str, idx: int = 0) -> torch.Tensor:
        npy_path = self.cache_index.get(v_path)
        if npy_path and os.path.exists(npy_path):
            arr = np.load(npy_path)  # (M, 224, 224, 3) uint8 RGB
            m = arr.shape[0]
            if m == 0:
                return torch.zeros((self.num_frames, 224, 224, 3), dtype=torch.uint8)

            if self.num_frames < m:
                if self.augment and not self.deterministic and m >= self.num_frames * 2:
                    offset = random.randint(0, (m // self.num_frames) - 1)
                    stride = m // self.num_frames
                    indices = np.arange(offset, m, stride)[:self.num_frames]
                else:
                    indices = np.linspace(0, m - 1, self.num_frames, dtype=int)
                arr = arr[indices]
            elif self.num_frames > m:
                indices = np.linspace(0, m - 1, self.num_frames, dtype=int)
                arr = arr[indices]

            if self.augment and not self.deterministic and random.random() < 0.5:
                arr = arr[:, :, ::-1, :]

            if self.domain_augment:
                if self.deterministic:
                    arr = _apply_clip_domain_degradation(arr, random.Random(10000 + int(idx)))
                elif random.random() < self.domain_prob:
                    arr = _apply_clip_domain_degradation(arr, random.Random(random.randint(0, 10_000_000)))

            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            return torch.from_numpy(arr)

        frames_tensor, _ = self.frame_extractor.extract_and_preprocess(v_path)
        return frames_tensor.squeeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        v_path = str(row["video_path"])
        label_str = str(row.get("label", "REAL")).upper()

        # Target & claim generation without empty-claim shortcut leakage
        if self.force_claim_mode == "deceptive_on_real":
            label = 2 if label_str == "REAL" else 1
            claim_text = random.choice(DECEPTIVE_SAMPLE_CLAIMS)
        elif self.force_claim_mode == "benign":
            label = 0 if label_str == "REAL" else 1
            claim_text = BENIGN_SAMPLE_CLAIMS[idx % len(BENIGN_SAMPLE_CLAIMS)]
        elif self.deterministic:
            label = 0 if label_str == "REAL" else 1
            claim_text = ""
        else:
            # Identical neutral caption distribution for Class 0 (REAL) and Class 1 (FAKE);
            # also include FAKE + deceptive claim -> Class 1 so visual forgery overrides clickbait text.
            neutral_claim = "" if random.random() < 0.6 else random.choice(BENIGN_SAMPLE_CLAIMS)
            if label_str == "REAL":
                if random.random() < self.contextual_misinfo_prob:
                    label = 2
                    claim_text = random.choice(DECEPTIVE_SAMPLE_CLAIMS)
                else:
                    label = 0
                    claim_text = neutral_claim
            else:
                label = 1
                claim_text = random.choice(DECEPTIVE_SAMPLE_CLAIMS) if random.random() < 0.12 else neutral_claim

        frames_tensor = self._load_frames_tensor(v_path, idx=idx)
        audio_tensor = torch.zeros(16000, dtype=torch.float32)
        text_tokens = tokenize_claim(claim_text)

        return {
            "frames": frames_tensor,
            "audio": audio_tensor,
            "tokens": text_tokens,
            "label": torch.tensor(label, dtype=torch.long),
            "binary_label": torch.tensor(0 if label_str == "REAL" else 1, dtype=torch.long),
            "category": str(row.get("category", label_str)),
            "video_path": v_path
        }


def collate_multimodal(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collator handling variable frame sequences and metadata.
    """
    labels = torch.stack([b["label"] for b in batch])
    binary_labels = torch.stack([b["binary_label"] for b in batch])
    audio = torch.stack([b["audio"] for b in batch])
    tokens = torch.stack([b["tokens"] for b in batch])

    max_n = max(b["frames"].shape[0] for b in batch)
    padded_frames = []
    for b in batch:
        f = b["frames"]
        n = f.shape[0]
        if n < max_n:
            pad_shape = (max_n - n,) + tuple(f.shape[1:])
            pad = torch.zeros(pad_shape, dtype=f.dtype)
            f_padded = torch.cat([f, pad], dim=0)
        else:
            f_padded = f
        padded_frames.append(f_padded)

    frames = torch.stack(padded_frames, dim=0)
    return {
        "frames": frames,
        "audio": audio,
        "tokens": tokens,
        "label": labels,
        "binary_label": binary_labels,
        "category": [b["category"] for b in batch],
    }


def make_balanced_sampler(manifest_df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Balances REAL vs all FAKE manipulation categories so each training epoch sees
    ~42% REAL clips and ~58% FAKE clips distributed equally across manipulation types.
    """
    cats = manifest_df.get("category", manifest_df["label"]).astype(str).tolist()
    labels = manifest_df["label"].astype(str).str.upper().tolist()

    cat_counts: Dict[str, int] = {}
    for c in cats:
        cat_counts[c] = cat_counts.get(c, 0) + 1

    fake_cats = [c for c, cnt in cat_counts.items() if c.lower() not in {"original", "real", "youtube", "actors"}]
    n_fake_cats = max(1, len(fake_cats))

    weights = []
    for c, lbl in zip(cats, labels):
        if lbl == "REAL":
            weights.append(0.42 / max(1, cat_counts.get(c, 1)))
        else:
            weights.append((0.58 / n_fake_cats) / max(1, cat_counts.get(c, 1)))

    return WeightedRandomSampler(weights=weights, num_samples=len(manifest_df), replacement=True)


def evaluate_visual_binary(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    use_amp: bool = True,
) -> Dict[str, Any]:
    """Evaluates binary deepfake detection (REAL vs FAKE with empty claim text) cleanly."""
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, balanced_accuracy_score

    model.eval()
    all_probs_3 = []
    all_logits_3 = []
    all_bin_labels = []
    all_cats = []

    with torch.no_grad():
        for batch in dataloader:
            frames = gpu_prepare_frames(batch["frames"], device, augment=False)
            audio = batch["audio"].to(device, non_blocking=True)
            tokens = batch["tokens"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(use_amp and str(device).startswith("cuda"))):
                logits, _ = model(frames, audio, tokens)
            logits_f = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
            probs = torch.softmax(logits_f, dim=-1)
            all_logits_3.append(logits_f.cpu().numpy())
            all_probs_3.append(probs.cpu().numpy())
            all_bin_labels.append(batch["binary_label"].numpy())
            all_cats.extend(batch["category"])

    logits_np = np.concatenate(all_logits_3, axis=0)
    probs_np = np.concatenate(all_probs_3, axis=0)
    y_true = np.concatenate(all_bin_labels, axis=0)

    # Normalized conditional P(Fake | Visual Payload in {Authentic, Manipulated})
    p_fake = probs_np[:, 1] / np.maximum(probs_np[:, 0] + probs_np[:, 1], 1e-8)
    y_pred = (p_fake >= 0.5).astype(int)

    acc = float((y_pred == y_true).mean())
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    roc_auc = float(roc_auc_score(y_true, p_fake)) if len(np.unique(y_true)) > 1 else 0.5
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    cat_acc: Dict[str, float] = {}
    cats_arr = np.array(all_cats)
    for cat in sorted(set(all_cats)):
        mask = cats_arr == cat
        cat_acc[cat] = round(float((y_pred[mask] == y_true[mask]).mean() * 100.0), 2)

    return {
        "accuracy_pct": round(acc * 100.0, 2),
        "balanced_accuracy_pct": round(bal_acc * 100.0, 2),
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
        "real_recall_pct": round(tn / max(1, tn + fp) * 100.0, 2),
        "fake_recall_pct": round(tp / max(1, tp + fn) * 100.0, 2),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "per_category_accuracy_pct": cat_acc,
        "p_fake": p_fake,
        "y_true": y_true,
        "logits_3": logits_np,
        "probs_3": probs_np,
    }


def train_fusion_model(
    manifest_path: Optional[str] = None,
    epochs: int = 8,
    batch_size: int = 16,
    limit_samples: Optional[int] = None,
    lr: float = 1.5e-4,
    device: Optional[str] = None,
    teacher_ckpt_path: Optional[Path] = None,
    num_frames: int = 8,
    domain_augment: bool = False,
    warm_start: bool = False,
) -> Dict[str, Any]:
    """
    Trains or fine-tunes the Distilled Edge Student model with Knowledge Distillation from
    ResNet-50 Teacher, saves the best validation checkpoint, and exports FP32/INT8 ONNX.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = str(dev).startswith("cuda")
    logger.info(f"Using compute device: {dev}")

    manifest_file = Path(manifest_path) if manifest_path else config.manifest_dir / "manifest.csv"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}. Run inspect_dataset first.")

    df = pd.read_csv(manifest_file)
    df["exists"] = df["video_path"].apply(os.path.exists)
    df = df[df["exists"]].reset_index(drop=True)
    logger.info(f"Loaded {len(df)} accessible videos from {manifest_file}")

    train_df = df[df["split"] == "train"].reset_index(drop=True) if "split" in df.columns else df
    val_df = df[df["split"] == "val"].reset_index(drop=True) if "split" in df.columns else train_df
    if train_df.empty:
        train_df = df
    if val_df.empty:
        val_df = train_df

    cache_index = load_frame_cache_index()
    train_ds = FaceForensicsMultimodalDataset(
        train_df, max_samples=limit_samples, cache_index=cache_index,
        num_frames=num_frames, augment=True, deterministic=False, domain_augment=domain_augment
    )
    val_ds = FaceForensicsMultimodalDataset(
        val_df, cache_index=cache_index,
        num_frames=num_frames, augment=False, deterministic=True
    )
    val_dom_ds = FaceForensicsMultimodalDataset(
        val_df, cache_index=cache_index,
        num_frames=num_frames, augment=False, deterministic=True, domain_augment=True
    ) if domain_augment else None

    nw = 0
    sampler = make_balanced_sampler(train_ds.df) if limit_samples is None else None
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None),
        collate_fn=collate_multimodal, num_workers=nw,
        pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_multimodal, num_workers=nw,
        pin_memory=use_cuda
    )
    val_dom_loader = DataLoader(
        val_dom_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_multimodal, num_workers=nw,
        pin_memory=use_cuda
    ) if val_dom_ds is not None else None

    model = VeriMediaStudentModel().to(dev)
    ckpt_path = config.checkpoints_dir / "verimedia_student_best.pth"
    if warm_start and ckpt_path.exists():
        model.load_state_dict(torch.load(str(ckpt_path), map_location=dev))
        logger.info(f"Warm-started Student model from {ckpt_path}")

    # Optional Teacher loading for genuine Knowledge Distillation
    teacher_model = None
    t_ckpt = teacher_ckpt_path or (config.checkpoints_dir / "verimedia_resnet50.pth")
    if t_ckpt.exists():
        try:
            from src.models_zoo import VeriMediaZooModel
            teacher_model = VeriMediaZooModel(backbone_type="resnet50", pretrained=False).to(dev)
            teacher_model.load_state_dict(torch.load(str(t_ckpt), map_location=dev))
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
            logger.info(f"Loaded Teacher model from {t_ckpt} for Knowledge Distillation.")
        except Exception as e:
            logger.warning(f"Could not load Teacher checkpoint ({e}); training Student standalone.")
            teacher_model = None

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)
    amp_dtype = torch.float16
    use_scaler = use_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    ce_weights = torch.tensor([1.15, 0.90, 1.80], dtype=torch.float32, device=dev)
    criterion = nn.CrossEntropyLoss(weight=ce_weights, label_smoothing=0.04)
    kd_loss_fn = DistillationLoss(alpha=0.5, temperature=2.0, discrepancy_weight=0.2)

    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            frames = gpu_prepare_frames(batch["frames"], dev, augment=True, domain_augment=domain_augment)
            audio = batch["audio"].to(dev, non_blocking=True)
            tokens = batch["tokens"].to(dev, non_blocking=True)
            labels = batch["label"].to(dev, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_cuda):
                logits, disc_dict = model(frames, audio, tokens)
                if teacher_model is not None:
                    with torch.no_grad():
                        t_logits, t_disc = teacher_model(frames, audio, tokens)
            if teacher_model is not None:
                total_loss, _ = kd_loss_fn(
                    logits.float(),
                    t_logits.float(),
                    labels,
                    {k: v.float() for k, v in disc_dict.items()},
                    {k: v.float() for k, v in t_disc.items()},
                )
            else:
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_cuda):
                    total_loss = criterion(logits, labels)

            if not torch.isfinite(total_loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            if use_scaler:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += total_loss.item() * len(labels)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        scheduler.step()
        avg_loss = epoch_loss / max(1, total)
        train_acc = (correct / max(1, total)) * 100.0
        val_res = evaluate_visual_binary(model, val_loader, dev, use_amp=use_cuda)
        val_dom_res = evaluate_visual_binary(model, val_dom_loader, dev, use_amp=use_cuda) if val_dom_loader is not None else None
        sel_auc = (0.40 * val_res["roc_auc"] + 0.60 * val_dom_res["roc_auc"]) if val_dom_res is not None else val_res["roc_auc"]

        row_hist = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_acc": val_res["accuracy_pct"],
            "val_bal_acc": val_res["balanced_accuracy_pct"],
            "val_roc_auc": val_res["roc_auc"],
            "val_f1": val_res["f1"],
        }
        if val_dom_res is not None:
            row_hist["val_dom_roc_auc"] = val_dom_res["roc_auc"]
            row_hist["val_dom_bal_acc"] = val_dom_res["balanced_accuracy_pct"]
        history.append(row_hist)
        dom_msg = f" | DomAUC={val_dom_res['roc_auc']:.4f} (RealRec={val_dom_res['real_recall_pct']:.1f}%)" if val_dom_res is not None else ""
        print(
            f"[Student] Ep {epoch}/{epochs} | Loss={avg_loss:.4f} | TrAcc={train_acc:.2f}% | "
            f"ValAcc={val_res['accuracy_pct']:.2f}% | ValBalAcc={val_res['balanced_accuracy_pct']:.2f}% | "
            f"ValAUC={val_res['roc_auc']:.4f}{dom_msg}",
            flush=True
        )

        if sel_auc >= best_auc:
            best_auc = sel_auc
            torch.save(model.state_dict(), str(ckpt_path))

    if ckpt_path.exists():
        model.load_state_dict(torch.load(str(ckpt_path), map_location=dev))

    model_cpu = model.cpu().eval()
    export_student_to_onnx(model_cpu)
    quant_info = quantize_onnx_to_int8()

    return {
        "final_epoch": epochs,
        "best_val_roc_auc": round(best_auc, 4),
        "history": history,
        "quantization_info": quant_info,
        "checkpoint_path": str(ckpt_path),
    }


def main():
    parser = argparse.ArgumentParser(description="PixelProof Late-Fusion Training & Distillation")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--limit_samples", type=int, default=None, help="Optional cap on training samples")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")

    args = parser.parse_args()
    res = train_fusion_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        limit_samples=args.limit_samples,
        lr=args.lr
    )
    print("\nTraining completed successfully:")
    print(res)


if __name__ == "__main__":
    main()
