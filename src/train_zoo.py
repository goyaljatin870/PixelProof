"""
train_zoo.py - VeriMedia Model Zoo GPU Training, Calibration, Ensemble Stacking & Test Evaluation
Trains all 6 specialist architectures (MobileNetV3, Temporal-LSTM, MesoInception-4,
ResNet-50, XceptionNet, EfficientNetV2-S), saves best .pth checkpoints, .pkl calibrators &
committee stacking ensemble, exports ONNX graphs, distills Edge Student + INT8 ONNX,
and evaluates everything on the identity-disjoint test split (1,092 videos).
"""

import os
import sys
import json
import time
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure package root is in sys.path
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, balanced_accuracy_score

from src.config import config
from src.models_zoo import VeriMediaZooModel
from src.fusion_model import VeriMediaEdgeONNXWrapper, VeriMediaStudentModel
from src.train import (
    FaceForensicsMultimodalDataset,
    collate_multimodal,
    load_frame_cache_index,
    make_balanced_sampler,
    evaluate_visual_binary,
    gpu_prepare_frames,
    train_fusion_model,
)

ALL_MODELS = [
    "mobilenet_v3",
    "temporal_lstm",
    "mesonet",
    "resnet50",
    "xception",
    "efficientnet_v2",
]

# Tuned per-architecture configs for 6 GB RTX 4050 dedicated VRAM (FP16 Tensor Cores, no shared-RAM spill)
ZOO_TRAIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "mobilenet_v3": {"batch_size": 16, "num_frames": 8, "grad_accum": 1, "head_lr": 2.0e-4, "backbone_lr": 1.0e-4, "pretrained": True, "amp": True},
    "temporal_lstm": {"batch_size": 16, "num_frames": 8, "grad_accum": 1, "head_lr": 2.0e-4, "backbone_lr": 1.0e-4, "pretrained": True, "amp": True},
    "mesonet": {"batch_size": 16, "num_frames": 8, "grad_accum": 1, "head_lr": 3.0e-4, "backbone_lr": 3.0e-4, "pretrained": False, "amp": False},
    "resnet50": {"batch_size": 4, "num_frames": 8, "grad_accum": 2, "head_lr": 1.5e-4, "backbone_lr": 7.5e-5, "pretrained": True, "amp": True},
    "xception": {"batch_size": 4, "num_frames": 4, "grad_accum": 2, "head_lr": 1.5e-4, "backbone_lr": 7.5e-5, "pretrained": True, "amp": True},
    "efficientnet_v2": {"batch_size": 4, "num_frames": 4, "grad_accum": 2, "head_lr": 1.5e-4, "backbone_lr": 7.5e-5, "pretrained": True, "amp": True},
}


def _find_optimal_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    best_thresh = 0.50
    best_bal = -1.0
    for thr in np.linspace(0.20, 0.85, 66):
        preds = (probs >= thr).astype(int)
        bal = balanced_accuracy_score(y_true, preds)
        if bal > best_bal:
            best_bal = bal
            best_thresh = float(thr)
    return round(best_thresh, 4)


def _score_binary_predictions(y_true: np.ndarray, probs: np.ndarray, threshold: float, categories: List[str]) -> Dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    acc = float((preds == y_true).mean() * 100.0)
    bal_acc = float(balanced_accuracy_score(y_true, preds) * 100.0)
    roc_auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.5
    f1 = float(f1_score(y_true, preds, zero_division=0))
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    cats_arr = np.array(categories)
    per_cat = {
        cat: round(float((preds[cats_arr == cat] == y_true[cats_arr == cat]).mean() * 100.0), 2)
        for cat in sorted(set(categories))
    }
    return {
        "threshold": round(float(threshold), 4),
        "accuracy_pct": round(acc, 2),
        "balanced_accuracy_pct": round(bal_acc, 2),
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
        "real_recall_pct": round(tn / max(1, tn + fp) * 100.0, 2),
        "fake_recall_pct": round(tp / max(1, tp + fn) * 100.0, 2),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "per_category_accuracy_pct": per_cat,
    }


def _evaluate_contextual_misinfo(
    model: nn.Module,
    val_real_df: pd.DataFrame,
    cache_index: Dict[str, str],
    num_frames: int,
    device: str,
    batch_size: int,
) -> Dict[str, Any]:
    if val_real_df.empty:
        return {"deceptive_recall_pct": 0.0, "benign_spec_pct": 0.0, "n": 0}

    use_cuda = str(device).startswith("cuda")
    dec_ds = FaceForensicsMultimodalDataset(
        val_real_df, cache_index=cache_index, num_frames=num_frames,
        augment=False, deterministic=True, force_claim_mode="deceptive_on_real"
    )
    ben_ds = FaceForensicsMultimodalDataset(
        val_real_df, cache_index=cache_index, num_frames=num_frames,
        augment=False, deterministic=True, force_claim_mode="benign"
    )
    dec_loader = DataLoader(dec_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal)
    ben_loader = DataLoader(ben_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal)

    model.eval()
    dec_preds, ben_preds = [], []
    with torch.no_grad():
        for b in dec_loader:
            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits, _ = model(gpu_prepare_frames(b["frames"], device, augment=False), b["audio"].to(device), b["tokens"].to(device))
            dec_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        for b in ben_loader:
            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits, _ = model(gpu_prepare_frames(b["frames"], device, augment=False), b["audio"].to(device), b["tokens"].to(device))
            ben_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    dec_recall = float(np.mean([p == 2 for p in dec_preds]) * 100.0) if dec_preds else 0.0
    ben_not_misinfo = float(np.mean([p == 0 for p in ben_preds]) * 100.0) if ben_preds else 0.0
    return {
        "deceptive_claim_class2_recall_pct": round(dec_recall, 2),
        "benign_claim_class0_acc_pct": round(ben_not_misinfo, 2),
        "n_real_clips": int(len(val_real_df)),
    }


def _load_clean_splits(limit_samples: Optional[int] = None) -> Dict[str, Any]:
    manifest_csv = config.manifest_dir / "manifest.csv"
    if not manifest_csv.exists():
        from src.inspect_dataset import build_manifest
        build_manifest()

    df = pd.read_csv(manifest_csv)
    df["exists"] = df["video_path"].apply(os.path.exists)
    df = df[df["exists"]].reset_index(drop=True)
    if df.empty:
        raise FileNotFoundError(f"Zero existing videos resolved from {manifest_csv}. Rebuild manifest first.")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    if train_df.empty:
        train_df = df
    if val_df.empty:
        val_df = train_df
    if test_df.empty:
        test_df = val_df

    if limit_samples is not None:
        train_df = train_df.sample(n=min(limit_samples, len(train_df)), random_state=42).reset_index(drop=True)

    return {"train": train_df, "val": val_df, "test": test_df}


def train_single_model(
    model_name: str,
    epochs: int = 8,
    limit_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    return_predictions: bool = False,
    domain_finetune: bool = False,
) -> Dict[str, Any]:
    """
    Trains (or domain-finetunes), validates, calibrates, exports, and evaluates a Zoo model.
    """
    t_model_start = time.perf_counter()
    model_name = model_name.lower()
    prof = ZOO_TRAIN_PROFILES.get(model_name, {"batch_size": 8, "num_frames": 8, "grad_accum": 1, "head_lr": 1.5e-4, "backbone_lr": 1.0e-4, "pretrained": True})
    bs = batch_size or prof["batch_size"]
    nf = prof["num_frames"]
    grad_accum = prof["grad_accum"]

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = str(dev).startswith("cuda")

    mode_str = "DOMAIN-ROBUST FINETUNE" if domain_finetune else "TRAIN"
    print(f"\n{'='*76}", flush=True)
    print(f"[*] {mode_str} {model_name.upper()} | epochs={epochs} | bs={bs} (accum={grad_accum}) | nf={nf} | dev={dev}", flush=True)
    print(f"{'='*76}", flush=True)

    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config.onnx_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    splits = _load_clean_splits(limit_samples=limit_samples)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    cache_index = load_frame_cache_index()

    train_ds = FaceForensicsMultimodalDataset(
        train_df, cache_index=cache_index, num_frames=nf, augment=True,
        deterministic=False, domain_augment=domain_finetune, domain_prob=(0.35 if domain_finetune else 0.0)
    )
    val_ds = FaceForensicsMultimodalDataset(
        val_df, cache_index=cache_index, num_frames=nf, augment=False, deterministic=True
    )
    test_ds = FaceForensicsMultimodalDataset(
        test_df, cache_index=cache_index, num_frames=nf, augment=False, deterministic=True
    )
    val_dom_ds = FaceForensicsMultimodalDataset(
        val_df, cache_index=cache_index, num_frames=nf, augment=False, deterministic=True, domain_augment=True
    ) if domain_finetune else None
    test_dom_ds = FaceForensicsMultimodalDataset(
        test_df, cache_index=cache_index, num_frames=nf, augment=False, deterministic=True, domain_augment=True
    ) if domain_finetune else None

    num_workers = 0
    eval_bs = min(16, bs * 2)
    sampler = make_balanced_sampler(train_df) if limit_samples is None else None
    train_loader = DataLoader(
        train_ds, batch_size=bs, sampler=sampler, shuffle=(sampler is None),
        collate_fn=collate_multimodal, num_workers=num_workers,
        pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_bs, shuffle=False,
        collate_fn=collate_multimodal, num_workers=num_workers,
        pin_memory=use_cuda
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_bs, shuffle=False,
        collate_fn=collate_multimodal, num_workers=num_workers,
        pin_memory=use_cuda
    )
    val_dom_loader = DataLoader(
        val_dom_ds, batch_size=eval_bs, shuffle=False,
        collate_fn=collate_multimodal, num_workers=num_workers,
        pin_memory=use_cuda
    ) if val_dom_ds is not None else None
    test_dom_loader = DataLoader(
        test_dom_ds, batch_size=eval_bs, shuffle=False,
        collate_fn=collate_multimodal, num_workers=num_workers,
        pin_memory=use_cuda
    ) if test_dom_ds is not None else None

    ckpt_path = config.checkpoints_dir / f"verimedia_{model_name}.pth"
    warm_from_ckpt = domain_finetune and ckpt_path.exists()
    model = VeriMediaZooModel(backbone_type=model_name, pretrained=(False if warm_from_ckpt else prof["pretrained"])).to(dev)
    if warm_from_ckpt:
        model.load_state_dict(torch.load(str(ckpt_path), map_location=dev))
        print(f"  [>] Warm-started from existing checkpoint: {ckpt_path.name}", flush=True)

    if warm_from_ckpt:
        for p in model.audio_backbone.parameters():
            p.requires_grad = False
        for p in model.nlp_backbone.parameters():
            p.requires_grad = False

    lr_mult = 0.20 if warm_from_ckpt else 1.0
    backbone_params = [p for p in model.visual_backbone.parameters() if p.requires_grad]
    head_params = (
        [p for p in model.audio_backbone.parameters() if p.requires_grad]
        + [p for p in model.nlp_backbone.parameters() if p.requires_grad]
        + [p for p in model.fusion_head.parameters() if p.requires_grad]
    )
    opt_groups = [{"params": backbone_params, "lr": prof["backbone_lr"] * lr_mult}]
    if head_params:
        opt_groups.append({"params": head_params, "lr": prof["head_lr"] * lr_mult})
    optimizer = optim.AdamW(opt_groups, weight_decay=1e-4)

    # 1-epoch linear warmup (fresh train) or smooth cosine decay (domain fine-tune)
    def lr_lambda(ep_idx: int) -> float:
        if not warm_from_ckpt and ep_idx == 0 and epochs > 2:
            return 0.5
        progress = float(ep_idx) / max(1, epochs - 1) if epochs > 1 else 0.0
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    use_amp = bool(use_cuda and prof.get("amp", True))
    amp_dtype = torch.float16
    use_scaler = use_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    ce_weights = torch.tensor([1.15, 0.90, 1.80], dtype=torch.float32, device=dev)
    criterion = nn.CrossEntropyLoss(weight=ce_weights, label_smoothing=0.04)

    best_score = -1.0
    best_val_res: Optional[Dict[str, Any]] = None
    best_val_dom_res: Optional[Dict[str, Any]] = None
    history = []

    if domain_finetune and val_dom_loader is not None:
        v0_clean = evaluate_visual_binary(model, val_loader, dev, use_amp=use_cuda)
        v0_dom = evaluate_visual_binary(model, val_dom_loader, dev, use_amp=use_cuda)
        print(
            f"  Ep 0 (Pre-Finetune) | CleanAUC={v0_clean['roc_auc']:.4f} Bal={v0_clean['balanced_accuracy_pct']:.1f}% | "
            f"CompressedAUC={v0_dom['roc_auc']:.4f} Bal={v0_dom['balanced_accuracy_pct']:.1f}% (RealRec={v0_dom['real_recall_pct']:.1f}%)",
            flush=True
        )

    for ep in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad(set_to_none=True)

        num_steps = len(train_loader)
        opt_stepped = False
        for step_idx, b in enumerate(train_loader):
            frames = gpu_prepare_frames(b["frames"], dev, augment=True, domain_augment=domain_finetune)
            audio = b["audio"].to(dev, non_blocking=True)
            tokens = b["tokens"].to(dev, non_blocking=True)
            labels = b["label"].to(dev, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(frames, audio, tokens)
                loss = criterion(logits.float(), labels)
                loss_scaled = loss / grad_accum

            if not torch.isfinite(loss_scaled):
                optimizer.zero_grad(set_to_none=True)
                continue

            if use_scaler:
                scaler.scale(loss_scaled).backward()
                if ((step_idx + 1) % grad_accum == 0) or ((step_idx + 1) == num_steps):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= old_scale:
                        opt_stepped = True
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss_scaled.backward()
                if ((step_idx + 1) % grad_accum == 0) or ((step_idx + 1) == num_steps):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    opt_stepped = True
                    optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item()) * len(labels)
            correct += (torch.argmax(logits, dim=-1) == labels).sum().item()
            total += len(labels)

        if opt_stepped or not use_cuda:
            scheduler.step()
        avg_loss = running_loss / max(1, total)
        tr_acc = correct / max(1, total) * 100.0

        val_res = evaluate_visual_binary(model, val_loader, dev, use_amp=use_cuda)
        val_dom_res = evaluate_visual_binary(model, val_dom_loader, dev, use_amp=use_cuda) if val_dom_loader is not None else None
        ep_sec = time.perf_counter() - ep_t0

        ep_record = {
            "epoch": ep,
            "train_loss": round(avg_loss, 4),
            "train_acc_pct": round(tr_acc, 2),
            "val_acc_pct": val_res["accuracy_pct"],
            "val_bal_acc_pct": val_res["balanced_accuracy_pct"],
            "val_real_recall_pct": val_res["real_recall_pct"],
            "val_fake_recall_pct": val_res["fake_recall_pct"],
            "val_roc_auc": val_res["roc_auc"],
            "val_f1": val_res["f1"],
            "epoch_sec": round(ep_sec, 1),
        }
        if val_dom_res is not None:
            ep_record["val_dom_roc_auc"] = val_dom_res["roc_auc"]
            ep_record["val_dom_bal_acc_pct"] = val_dom_res["balanced_accuracy_pct"]
            ep_record["val_dom_real_recall_pct"] = val_dom_res["real_recall_pct"]
            score = (
                0.45 * val_res["roc_auc"]
                + 0.55 * val_dom_res["roc_auc"]
                + 0.001 * (0.45 * val_res["balanced_accuracy_pct"] + 0.55 * val_dom_res["balanced_accuracy_pct"])
            )
        else:
            score = val_res["roc_auc"] + 0.001 * val_res["balanced_accuracy_pct"]
        history.append(ep_record)

        improved = score > best_score
        if improved:
            best_score = score
            best_val_res = val_res
            best_val_dom_res = val_dom_res
            torch.save(model.state_dict(), str(ckpt_path))

        dom_msg = (
            f" | DomAUC={val_dom_res['roc_auc']:.4f} DomBal={val_dom_res['balanced_accuracy_pct']:5.2f}% (R={val_dom_res['real_recall_pct']:4.1f}%)"
            if val_dom_res is not None else ""
        )
        print(
            f"  Ep {ep}/{epochs} ({ep_sec:4.1f}s) | Loss={avg_loss:.4f} | TrAcc={tr_acc:5.2f}% | "
            f"ValAUC={val_res['roc_auc']:.4f} | ValAcc={val_res['accuracy_pct']:5.2f}% | "
            f"BalAcc={val_res['balanced_accuracy_pct']:5.2f}% (R={val_res['real_recall_pct']:4.1f}%, F={val_res['fake_recall_pct']:4.1f}%){dom_msg}"
            + (" [BEST]" if improved else ""),
            flush=True
        )

    # Reload best validation checkpoint before calibrating & testing
    model.load_state_dict(torch.load(str(ckpt_path), map_location=dev))
    val_best = best_val_res if best_val_res is not None else evaluate_visual_binary(model, val_loader, dev, use_amp=use_cuda)
    val_dom_best = best_val_dom_res if best_val_dom_res is not None else (
        evaluate_visual_binary(model, val_dom_loader, dev, use_amp=use_cuda) if val_dom_loader is not None else None
    )
    test_best = evaluate_visual_binary(model, test_loader, dev, use_amp=use_cuda)
    test_dom_best = evaluate_visual_binary(model, test_dom_loader, dev, use_amp=use_cuda) if test_dom_loader is not None else None

    # Fit Platt logistic calibrator on clean (or clean+compressed) validation predictions
    val_p = val_best["p_fake"]
    val_y = val_best["y_true"]
    if val_dom_best is not None:
        val_p_fit = np.concatenate([val_p, val_dom_best["p_fake"]], axis=0)
        val_y_fit = np.concatenate([val_y, val_dom_best["y_true"]], axis=0)
    else:
        val_p_fit, val_y_fit = val_p, val_y

    test_p = test_best["p_fake"]
    test_y = test_best["y_true"]

    calibrator = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    calibrator.fit(val_p_fit.reshape(-1, 1), val_y_fit)
    val_p_cal = calibrator.predict_proba(val_p_fit.reshape(-1, 1))[:, 1]
    test_p_cal = calibrator.predict_proba(test_p.reshape(-1, 1))[:, 1]

    opt_thresh = _find_optimal_threshold(val_y_fit, val_p_cal)
    test_cats = test_df["category"].astype(str).tolist()
    val_cats = val_df["category"].astype(str).tolist()

    val_metrics_raw = _score_binary_predictions(val_y, val_p, 0.50, val_cats)
    test_metrics_raw = _score_binary_predictions(test_y, test_p, 0.50, test_cats)
    test_metrics_cal = _score_binary_predictions(test_y, test_p_cal, opt_thresh, test_cats)
    test_domain_metrics = None
    if test_dom_best is not None:
        td_p = test_dom_best["p_fake"]
        td_y = test_dom_best["y_true"]
        td_p_cal = calibrator.predict_proba(td_p.reshape(-1, 1))[:, 1]
        test_domain_metrics = {
            "raw_05": _score_binary_predictions(td_y, td_p, 0.50, test_cats),
            "calibrated": _score_binary_predictions(td_y, td_p_cal, opt_thresh, test_cats),
        }

    test_real_df = test_df[test_df["label"].astype(str).str.upper() == "REAL"].reset_index(drop=True)
    ctx_metrics = _evaluate_contextual_misinfo(model, test_real_df, cache_index, nf, dev, bs)

    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

    # Export per-model ONNX graph on CPU
    onnx_path = config.onnx_dir / f"verimedia_{model_name}.onnx"
    onnx_size_mb = 0.0
    try:
        model_cpu = model.cpu().eval()
        export_wrapper = VeriMediaEdgeONNXWrapper(model_cpu.fusion_head).eval()
        dummy_v = torch.randn(1, model_cpu.visual_dim, dtype=torch.float32)
        dummy_a = torch.randn(1, model_cpu.audio_dim, dtype=torch.float32)
        dummy_t = torch.randn(1, model_cpu.text_dim, dtype=torch.float32)
        torch.onnx.export(
            export_wrapper,
            (dummy_v, dummy_a, dummy_t),
            str(onnx_path),
            export_params=True,
            opset_version=18,
            input_names=["visual_features", "audio_features", "text_features"],
            output_names=["probabilities"],
            dynamo=False,
        )
        onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    except Exception as e:
        print(f"  [-] ONNX export warning for {model_name}: {e}", flush=True)

    # Save calibrated .pkl artifact bundle
    pkl_path = config.checkpoints_dir / f"verimedia_{model_name}_calibrator.pkl"
    pkl_bundle = {
        "model_name": model_name,
        "checkpoint_file": str(ckpt_path),
        "onnx_file": str(onnx_path),
        "profile": prof,
        "epochs": epochs,
        "domain_finetuned": bool(domain_finetune),
        "calibrator": calibrator,
        "optimal_threshold": opt_thresh,
        "history": history,
        "val_metrics": val_metrics_raw,
        "test_metrics_raw_05": test_metrics_raw,
        "test_metrics_calibrated": test_metrics_cal,
        "test_domain_metrics": test_domain_metrics,
        "contextual_misinfo_test": ctx_metrics,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(pkl_bundle, f)

    elapsed_min = (time.perf_counter() - t_model_start) / 60.0
    dom_test_str = (
        f" | DomTestAUC={test_domain_metrics['raw_05']['roc_auc']:.4f} CalAcc={test_domain_metrics['calibrated']['accuracy_pct']}%"
        if test_domain_metrics is not None else ""
    )
    print(
        f"  [+] FINISHED {model_name.upper()} in {elapsed_min:.2f} min | "
        f"Ckpt={ckpt_path.name} ({ckpt_size_mb:.2f} MB) | Pkl={pkl_path.name}\n"
        f"      TEST (1,092 vids) -> ROC-AUC={test_metrics_raw['roc_auc']:.4f} | "
        f"RawAcc@0.5={test_metrics_raw['accuracy_pct']}% (Bal={test_metrics_raw['balanced_accuracy_pct']}%) | "
        f"CalAcc@{opt_thresh:.2f}={test_metrics_cal['accuracy_pct']}% (Bal={test_metrics_cal['balanced_accuracy_pct']}%) | "
        f"CM={test_metrics_cal['confusion_matrix']} | CtxClass2Recall={ctx_metrics['deceptive_claim_class2_recall_pct']}%{dom_test_str}",
        flush=True
    )

    import gc
    del model
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    res_summary: Dict[str, Any] = {
        "model_name": model_name,
        "checkpoint_file": str(ckpt_path),
        "checkpoint_size_mb": round(ckpt_size_mb, 2),
        "calibrator_pkl_file": str(pkl_path),
        "onnx_file": str(onnx_path),
        "onnx_size_mb": round(onnx_size_mb, 2),
        "train_minutes": round(elapsed_min, 2),
        "domain_finetuned": bool(domain_finetune),
        "history": history,
        "val_metrics_raw": val_metrics_raw,
        "test_metrics_raw": test_metrics_raw,
        "test_metrics_calibrated": test_metrics_cal,
        "test_domain_metrics": test_domain_metrics,
        "contextual_misinfo_test": ctx_metrics,
    }
    if return_predictions:
        res_summary["_val_p_fake"] = val_p_fit
        res_summary["_val_y"] = val_y_fit
        res_summary["_test_p_fake"] = test_p
        res_summary["_test_y"] = test_y
        res_summary["_test_cats"] = test_cats
        if test_dom_best is not None:
            res_summary["_test_dom_p_fake"] = test_dom_best["p_fake"]
            res_summary["_test_dom_y"] = test_dom_best["y_true"]
    return res_summary


def _eval_completed_model_predictions(
    model_name: str,
    limit_samples: Optional[int] = None,
    device: Optional[str] = None,
    domain_eval: bool = False,
) -> Dict[str, Any]:
    """Re-computes val/test probabilities from an already-completed .pth checkpoint for ensemble stacking."""
    prof = ZOO_TRAIN_PROFILES.get(model_name, {"batch_size": 8, "num_frames": 8})
    nf = prof["num_frames"]
    eval_bs = min(16, prof["batch_size"] * 2)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = str(device or dev).startswith("cuda")

    splits = _load_clean_splits(limit_samples=limit_samples)
    val_df, test_df = splits["val"], splits["test"]
    cache_index = load_frame_cache_index()
    val_ds = FaceForensicsMultimodalDataset(val_df, cache_index=cache_index, num_frames=nf, augment=False, deterministic=True)
    test_ds = FaceForensicsMultimodalDataset(test_df, cache_index=cache_index, num_frames=nf, augment=False, deterministic=True)
    val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False, collate_fn=collate_multimodal, pin_memory=use_cuda)
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, collate_fn=collate_multimodal, pin_memory=use_cuda)

    ckpt_path = config.checkpoints_dir / f"verimedia_{model_name}.pth"
    model = VeriMediaZooModel(backbone_type=model_name, pretrained=False).to(dev)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=dev))
    val_res = evaluate_visual_binary(model, val_loader, dev, use_amp=use_cuda)
    test_res = evaluate_visual_binary(model, test_loader, dev, use_amp=use_cuda)

    val_p_fit, val_y_fit = val_res["p_fake"], val_res["y_true"]
    test_dom_p, test_dom_y = None, None
    if domain_eval:
        val_dom_ds = FaceForensicsMultimodalDataset(val_df, cache_index=cache_index, num_frames=nf, augment=False, deterministic=True, domain_augment=True)
        test_dom_ds = FaceForensicsMultimodalDataset(test_df, cache_index=cache_index, num_frames=nf, augment=False, deterministic=True, domain_augment=True)
        val_dom_loader = DataLoader(val_dom_ds, batch_size=eval_bs, shuffle=False, collate_fn=collate_multimodal, pin_memory=use_cuda)
        test_dom_loader = DataLoader(test_dom_ds, batch_size=eval_bs, shuffle=False, collate_fn=collate_multimodal, pin_memory=use_cuda)
        val_dom_res = evaluate_visual_binary(model, val_dom_loader, dev, use_amp=use_cuda)
        test_dom_res = evaluate_visual_binary(model, test_dom_loader, dev, use_amp=use_cuda)
        val_p_fit = np.concatenate([val_p_fit, val_dom_res["p_fake"]], axis=0)
        val_y_fit = np.concatenate([val_y_fit, val_dom_res["y_true"]], axis=0)
        test_dom_p, test_dom_y = test_dom_res["p_fake"], test_dom_res["y_true"]

    import gc
    del model
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    return {
        "val_p_fake": val_p_fit,
        "val_y": val_y_fit,
        "test_p_fake": test_res["p_fake"],
        "test_y": test_res["y_true"],
        "test_cats": test_df["category"].astype(str).tolist(),
        "test_dom_p_fake": test_dom_p,
        "test_dom_y": test_dom_y,
    }


def train_all_models(
    epochs: int = 8,
    limit_samples: Optional[int] = None,
    models_list: Optional[List[str]] = None,
    resume: bool = True,
    domain_finetune: bool = False,
) -> Dict[str, Any]:
    """
    Trains (or domain-finetunes) all 6 Model Zoo backbones, fits calibrated Committee Stacking (.pkl),
    runs Knowledge Distillation into Edge Student (.pth + FP32/INT8 ONNX), and saves full test reports.
    """
    t_all = time.perf_counter()
    target_models = models_list or ALL_MODELS
    results: Dict[str, Any] = {}
    val_probs_by_model: Dict[str, np.ndarray] = {}
    test_probs_by_model: Dict[str, np.ndarray] = {}
    test_dom_probs_by_model: Dict[str, np.ndarray] = {}
    val_y_ref = None
    test_y_ref = None
    test_dom_y_ref = None
    test_cats_ref = None

    report_json = config.reports_dir / "zoo_training_report.json"
    report_pkl = config.reports_dir / "zoo_training_report.pkl"

    existing_models: Dict[str, Any] = {}
    if resume and report_json.exists():
        try:
            with open(report_json, "r", encoding="utf-8") as f:
                existing_models = json.load(f).get("models", {})
        except Exception:
            existing_models = {}

    for m in target_models:
        ckpt_file = config.checkpoints_dir / f"verimedia_{m}.pth"
        pkl_file = config.checkpoints_dir / f"verimedia_{m}_calibrator.pkl"
        prev_rec = existing_models.get(m)
        same_mode = (bool(prev_rec.get("domain_finetuned", False)) == bool(domain_finetune)) if prev_rec else False
        if (
            resume
            and prev_rec is not None
            and same_mode
            and len(prev_rec.get("history", [])) == epochs
            and ckpt_file.exists()
            and pkl_file.exists()
        ):
            print(f"\n[*] Resuming: {m.upper()} already completed {epochs}/{epochs} epochs -> scoring saved checkpoint...", flush=True)
            preds = _eval_completed_model_predictions(m, limit_samples=limit_samples, domain_eval=domain_finetune)
            val_probs_by_model[m] = preds["val_p_fake"]
            test_probs_by_model[m] = preds["test_p_fake"]
            if preds.get("test_dom_p_fake") is not None:
                test_dom_probs_by_model[m] = preds["test_dom_p_fake"]
                test_dom_y_ref = preds["test_dom_y"]
            val_y_ref = preds["val_y"]
            test_y_ref = preds["test_y"]
            test_cats_ref = preds["test_cats"]
            results[m] = prev_rec
            continue

        res = train_single_model(m, epochs=epochs, limit_samples=limit_samples, return_predictions=True, domain_finetune=domain_finetune)
        val_probs_by_model[m] = res.pop("_val_p_fake")
        test_probs_by_model[m] = res.pop("_test_p_fake")
        val_y_ref = res.pop("_val_y")
        test_y_ref = res.pop("_test_y")
        test_cats_ref = res.pop("_test_cats")
        if "_test_dom_p_fake" in res:
            test_dom_probs_by_model[m] = res.pop("_test_dom_p_fake")
            test_dom_y_ref = res.pop("_test_dom_y")
        results[m] = res

        # Flush incremental report after each completed model
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump({"domain_finetuned": bool(domain_finetune), "models": results}, f, indent=2)

    # Committee Ensemble (Soft-Vote + Validation-Trained Logistic Regression Stacker .pkl)
    ensemble_report: Dict[str, Any] = {}
    if len(target_models) >= 2 and val_y_ref is not None and test_y_ref is not None:
        X_val = np.column_stack([val_probs_by_model[m] for m in target_models])
        X_test = np.column_stack([test_probs_by_model[m] for m in target_models])

        # Unweighted committee average
        soft_val = X_val.mean(axis=1)
        soft_test = X_test.mean(axis=1)
        soft_thresh = _find_optimal_threshold(val_y_ref, soft_val)
        soft_test_metrics = _score_binary_predictions(test_y_ref, soft_test, soft_thresh, test_cats_ref)
        soft_test_raw = _score_binary_predictions(test_y_ref, soft_test, 0.50, test_cats_ref)

        # Weighted stacking meta-learner (learned weights per committee specialist)
        meta_clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
        meta_clf.fit(X_val, val_y_ref)
        stack_val = meta_clf.predict_proba(X_val)[:, 1]
        stack_test = meta_clf.predict_proba(X_test)[:, 1]
        stack_thresh = _find_optimal_threshold(val_y_ref, stack_val)
        stack_test_metrics = _score_binary_predictions(test_y_ref, stack_test, stack_thresh, test_cats_ref)
        stack_test_raw = _score_binary_predictions(test_y_ref, stack_test, 0.50, test_cats_ref)

        dom_ens_metrics = None
        if len(test_dom_probs_by_model) == len(target_models) and test_dom_y_ref is not None:
            X_dom = np.column_stack([test_dom_probs_by_model[m] for m in target_models])
            stack_dom = meta_clf.predict_proba(X_dom)[:, 1]
            dom_ens_metrics = {
                "stacked_dom_raw_05": _score_binary_predictions(test_dom_y_ref, stack_dom, 0.50, test_cats_ref),
                "stacked_dom_calibrated": _score_binary_predictions(test_dom_y_ref, stack_dom, stack_thresh, test_cats_ref),
            }

        ens_pkl_path = config.checkpoints_dir / "verimedia_committee_ensemble.pkl"
        with open(ens_pkl_path, "wb") as f:
            pickle.dump({
                "model_names": target_models,
                "meta_classifier": meta_clf,
                "coefficients": {m: float(w) for m, w in zip(target_models, meta_clf.coef_[0])},
                "intercept": float(meta_clf.intercept_[0]),
                "optimal_threshold": stack_thresh,
                "soft_vote_threshold": soft_thresh,
                "soft_vote_test": soft_test_metrics,
                "stacked_test": stack_test_metrics,
                "domain_finetuned": bool(domain_finetune),
            }, f)

        ensemble_report = {
            "ensemble_pkl_file": str(ens_pkl_path),
            "specialist_weights": {m: round(float(w), 4) for m, w in zip(target_models, meta_clf.coef_[0])},
            "soft_vote_raw_05": soft_test_raw,
            "soft_vote_calibrated": soft_test_metrics,
            "stacked_committee_raw_05": stack_test_raw,
            "stacked_committee_calibrated": stack_test_metrics,
            "domain_test": dom_ens_metrics,
        }
        print(
            f"\n[Committee Ensemble] SoftVote AUC={soft_test_metrics['roc_auc']:.4f} Acc={soft_test_metrics['accuracy_pct']}% | "
            f"Stacked (.pkl) AUC={stack_test_metrics['roc_auc']:.4f} Acc={stack_test_metrics['accuracy_pct']}% BalAcc={stack_test_metrics['balanced_accuracy_pct']}%",
            flush=True
        )

    # Distilled Edge Student + INT8 ONNX export
    print(f"\n{'='*76}", flush=True)
    print("[*] Running Knowledge Distillation -> Edge Student (verimedia_student_best.pth + INT8 ONNX)", flush=True)
    print(f"{'='*76}", flush=True)
    student_ckpt_exists = (config.checkpoints_dir / "verimedia_student_best.pth").exists()
    student_res = train_fusion_model(
        epochs=epochs,
        batch_size=16,
        limit_samples=limit_samples,
        lr=(5.0e-5 if (domain_finetune and student_ckpt_exists) else 1.5e-4),
        teacher_ckpt_path=config.checkpoints_dir / "verimedia_resnet50.pth",
        domain_augment=domain_finetune,
        warm_start=(domain_finetune and student_ckpt_exists),
    )

    # Evaluate distilled Student cleanly on untouched test split (1,092 videos)
    splits = _load_clean_splits(limit_samples=limit_samples)
    test_df = splits["test"]
    cache_index = load_frame_cache_index()
    test_ds = FaceForensicsMultimodalDataset(test_df, cache_index=cache_index, num_frames=8, augment=False, deterministic=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_multimodal)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    student_model = VeriMediaStudentModel().to(dev)
    student_model.load_state_dict(torch.load(str(config.checkpoints_dir / "verimedia_student_best.pth"), map_location=dev))
    student_test_res = evaluate_visual_binary(student_model, test_loader, dev, use_amp=str(dev).startswith("cuda"))
    student_res["test_metrics"] = {
        k: v for k, v in student_test_res.items() if k not in {"p_fake", "y_true", "logits_3", "probs_3"}
    }

    total_minutes = round((time.perf_counter() - t_all) / 60.0, 2)
    full_report = {
        "total_minutes": total_minutes,
        "epochs": epochs,
        "domain_finetuned": bool(domain_finetune),
        "models": results,
        "committee_ensemble": ensemble_report,
        "distilled_student": student_res,
    }
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    with open(report_pkl, "wb") as f:
        pickle.dump(full_report, f)

    print(f"\n[+] ALL 6 ZOO MODELS + ENSEMBLE + DISTILLED STUDENT COMPLETED IN {total_minutes:.2f} MIN!", flush=True)
    return full_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Models in VeriMedia Model Zoo")
    parser.add_argument("--model", type=str, default="all", help="Model name (or 'all')")
    parser.add_argument("--epochs", type=int, default=8, help="Epochs per model")
    parser.add_argument("--samples", type=int, default=None, help="Optional cap on training samples")
    parser.add_argument("--no_resume", action="store_true", help="Do not skip already completed models")
    parser.add_argument("--domain_finetune", action="store_true", help="Warm-start domain-robust compression fine-tuning")

    args = parser.parse_args()
    if args.model == "all":
        train_all_models(
            epochs=args.epochs,
            limit_samples=args.samples,
            resume=(not args.no_resume),
            domain_finetune=args.domain_finetune,
        )
    else:
        train_single_model(
            args.model,
            epochs=args.epochs,
            limit_samples=args.samples,
            domain_finetune=args.domain_finetune,
        )
