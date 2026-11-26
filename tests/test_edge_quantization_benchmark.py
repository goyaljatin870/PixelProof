"""
Unit tests for ONNX Export, Dynamic INT8 Quantization, and System Verification Matrix Benchmark.
"""

from src.quantization import export_student_to_onnx, quantize_onnx_to_int8
from src.benchmark import benchmark_edge_model
from src.config import config


def test_onnx_export_and_int8_quantization():
    q_res = quantize_onnx_to_int8()
    assert q_res["is_size_compliant"], f"Model size {q_res['int8_size_mb']}MB exceeded target 45MB"
    assert q_res["int8_size_mb"] <= 45.0
    assert q_res["compression_ratio"] > 1.5
    assert q_res["test_latency_ms"] < 300.0


def test_system_verification_benchmark():
    report = benchmark_edge_model(num_iterations=10, warmup_runs=2)
    assert report["status"] == "PASS"
    m = report["metrics"]
    assert m["model_size"]["compliant"]
    assert m["inference_latency"]["compliant"]
    assert m["active_memory_footprint"]["compliant"]
