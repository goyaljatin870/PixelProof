"""
quantization.py - ONNX Export & 8-Bit Dynamic Quantization Engine
Converts PyTorch student models to ONNX runtime format and quantizes weights to INT8,
producing an edge-compliant model binary <= 45 MB with sub-300ms latency.
"""

import os
import sys
import time

# Reconfigure stdout/stderr to utf-8 to prevent Windows cp1252 charmap errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

from src.fusion_model import VeriMediaStudentModel, VeriMediaEdgeONNXWrapper
from src.config import config


def export_student_to_onnx(
    student_model: Optional[nn.Module] = None,
    output_fp32_path: Optional[str] = None
) -> Tuple[str, float]:
    """
    Exports distilled PyTorch student model to standard FP32 ONNX runtime graph.
    """
    config.onnx_dir.mkdir(parents=True, exist_ok=True)
    if output_fp32_path is None:
        output_fp32_path = str(config.onnx_dir / "verimedia_student_fp32.onnx")

    if student_model is None:
        student_model = VeriMediaStudentModel()
        ckpt = config.checkpoints_dir / "verimedia_student_best.pth"
        if ckpt.exists():
            student_model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))

    student_model.eval()
    
    # Prepare wrapper operating on multimodal representations for edge deployment
    export_wrapper = VeriMediaEdgeONNXWrapper(student_model.fusion_head)
    export_wrapper.eval()

    dummy_v = torch.randn(1, student_model.visual_dim, dtype=torch.float32)
    dummy_a = torch.randn(1, student_model.audio_dim, dtype=torch.float32)
    dummy_t = torch.randn(1, student_model.text_dim, dtype=torch.float32)

    # Use dynamo=False for deterministic, ultra-fast TorchScript-based ONNX export
    torch.onnx.export(
        export_wrapper,
        (dummy_v, dummy_a, dummy_t),
        output_fp32_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["visual_features", "audio_features", "text_features"],
        output_names=["probabilities"],
        dynamic_axes={
            "visual_features": {0: "batch_size"},
            "audio_features": {0: "batch_size"},
            "text_features": {0: "batch_size"},
            "probabilities": {0: "batch_size"}
        },
        dynamo=False
    )

    size_mb = os.path.getsize(output_fp32_path) / (1024 * 1024)
    return output_fp32_path, round(size_mb, 2)


def quantize_onnx_to_int8(
    input_fp32_path: Optional[str] = None,
    output_int8_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quantizes FP32 ONNX model to 8-bit integers (QUInt8 / QInt8) for edge deployment.
    Enforces VeriMedia System Verification Matrix requirement: Model Size <= 45 MB.
    """
    config.onnx_dir.mkdir(parents=True, exist_ok=True)
    
    if input_fp32_path is None:
        input_fp32_path = str(config.onnx_dir / "verimedia_student_fp32.onnx")
        ckpt = config.checkpoints_dir / "verimedia_student_best.pth"
        needs_export = (not os.path.exists(input_fp32_path)) or (
            ckpt.exists() and ckpt.stat().st_mtime > os.path.getmtime(input_fp32_path)
        )
        if needs_export:
            export_student_to_onnx(output_fp32_path=input_fp32_path)

    if output_int8_path is None:
        output_int8_path = str(config.onnx_dir / "verimedia_edge_int8.onnx")

    # Dynamic Post-Training Quantization (PTQ) via ONNX Runtime
    quantize_dynamic(
        model_input=input_fp32_path,
        model_output=output_int8_path,
        weight_type=QuantType.QUInt8
    )

    fp32_size_mb = os.path.getsize(input_fp32_path) / (1024 * 1024)
    int8_size_mb = os.path.getsize(output_int8_path) / (1024 * 1024)
    comp_ratio = fp32_size_mb / max(1e-5, int8_size_mb)

    # Verify edge specification constraint (<= 45 MB)
    is_compliant = int8_size_mb <= config.edge.max_model_size_mb

    # Verify execution with ONNX Runtime
    session = ort.InferenceSession(output_int8_path, providers=["CPUExecutionProvider"])
    
    # Warmup and test inference
    dummy_v = torch.randn(1, config.model.student_visual_dim).numpy()
    dummy_a = torch.randn(1, config.model.audio_feature_dim).numpy()
    dummy_t = torch.randn(1, config.model.text_feature_dim).numpy()

    t0 = time.perf_counter()
    outputs = session.run(None, {
        "visual_features": dummy_v,
        "audio_features": dummy_a,
        "text_features": dummy_t
    })
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "fp32_model_path": input_fp32_path,
        "fp32_size_mb": round(fp32_size_mb, 2),
        "int8_model_path": output_int8_path,
        "int8_size_mb": round(int8_size_mb, 2),
        "compression_ratio": round(comp_ratio, 2),
        "target_max_size_mb": config.edge.max_model_size_mb,
        "is_size_compliant": is_compliant,
        "test_latency_ms": round(elapsed_ms, 2),
        "test_output_shape": [list(o.shape) for o in outputs]
    }
