"""
cli.py - VeriMedia Command Line Interface
Unified CLI for Dataset Inspection, WhatsApp Degradation, ONNX Quantization,
System Verification Matrix Benchmarking, and Media Prediction.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Ensure package root is in sys.path
_pkg_root = Path(__file__).resolve().parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from src.config import config
from src.inspect_dataset import build_manifest, print_manifest_report
from src.degradation_pipeline import degrade_video_whatsapp, batch_degrade_dataset
from src.quantization import quantize_onnx_to_int8
from src.benchmark import benchmark_edge_model
from src.pipeline import VeriMediaEngine


def cmd_inspect(args):
    print(f"[*] Scanning FaceForensics++ at: {args.dataset_path}")
    df, summary = build_manifest(ff_path=args.dataset_path)
    print_manifest_report(summary)


def cmd_degrade(args):
    print(f"[*] Simulating WhatsApp compression on: {args.video_path}")
    res = degrade_video_whatsapp(args.video_path, crf=args.crf, target_fps=args.fps)
    if res.get("success"):
        print("\n[+] Degradation Successful!")
        print(f"  - Input: {res['input_path']} ({res['orig_size_bytes'] / 1024:.1f} KB)")
        print(f"  - Degraded: {res['output_path']} ({res['degraded_size_bytes'] / 1024:.1f} KB)")
        print(f"  - Compression Ratio: {res['compression_ratio']}x reduction")
        print(f"  - Quantization CRF: {res['crf']} (Target FPS: {res['fps']})")
    else:
        print(f"\n[-] Degradation Failed: {res.get('error')}")


def cmd_quantize(args):
    print("[*] Exporting distilled student model to ONNX and applying INT8 Dynamic Quantization...")
    res = quantize_onnx_to_int8()
    print("\n[+] Quantization Complete:")
    print(f"  - FP32 ONNX Model: {res['fp32_model_path']} ({res['fp32_size_mb']} MB)")
    print(f"  - INT8 Edge Binary: {res['int8_model_path']} ({res['int8_size_mb']} MB)")
    print(f"  - Compression Ratio: {res['compression_ratio']}x")
    print(f"  - Compliance with <= 45MB Limit: {'YES ✅' if res['is_size_compliant'] else 'NO ❌'}")
    print(f"  - Test Inference Latency: {res['test_latency_ms']} ms")


def cmd_benchmark(args):
    print("[*] Running System Verification Matrix Benchmarking Protocol...")
    report = benchmark_edge_model(num_iterations=args.iterations)
    print("\n" + "="*65)
    print(f"SYSTEM VERIFICATION MATRIX STATUS: {report['status']}")
    print("="*65)
    m = report["metrics"]
    print(f"1. Model Binary Size:  {m['model_size']['value_mb']} MB (Target <= {m['model_size']['target_mb']} MB) -> {'PASS ✅' if m['model_size']['compliant'] else 'FAIL ❌'}")
    print(f"2. Inference Latency:  {m['inference_latency']['mean_ms']} ms (Target < {m['inference_latency']['target_ms']} ms) -> {'PASS ✅' if m['inference_latency']['compliant'] else 'FAIL ❌'}")
    print(f"   - p50: {m['inference_latency']['p50_ms']} ms | p95: {m['inference_latency']['p95_ms']} ms | min: {m['inference_latency']['min_ms']} ms")
    print(f"3. Active RAM Footprint: {m['active_memory_footprint']['active_delta_mb']} MB (Target < {m['active_memory_footprint']['target_mb']} MB) -> {'PASS ✅' if m['active_memory_footprint']['compliant'] else 'FAIL ❌'}")
    print(f"Report saved to: {config.reports_dir / 'system_verification_matrix.md'}")
    print("="*65 + "\n")


def cmd_predict(args):
    use_committee = getattr(args, "committee", False)
    if use_committee:
        from src.pipeline import ForensicCommitteeEngine
        print(f"[*] Analyzing Media Payload across Forensic Committee Ensemble: {args.video_path}")
        if args.claim:
            print(f"[*] Accompanying Claim / Caption: \"{args.claim}\"")
        if args.degrade:
            print("[*] Pre-processing with WhatsApp H.264 CRF 38 degradation simulator...")

        engine = ForensicCommitteeEngine()
        result = engine.verify_media_committee(
            video_path=args.video_path,
            claim_text=args.claim,
            simulate_whatsapp_degradation=args.degrade
        )

        c = result["consensus"]
        print("\n" + "="*65)
        print("VERIMEDIA FORENSIC COMMITTEE CONSENSUS REPORT")
        print("="*65)
        print(f"MASTER CONSENSUS VERDICT: {c['verdict'].upper()} (Confidence: {c['confidence']}%)")
        print(f"COMMITTEE AGREEMENT: {c['agreement_count']} of {c['total_models']} specialists ({c['agreement_pct']}%)")
        if c.get("has_dissent"):
            print(f"⚠️ SPECIALIST DISSENT: {', '.join(c.get('dissenting_specialists', []))}")

        print("\n[Autonomous Specialist Committee Verdicts]")
        for m_id, m_rep in result["committee"].items():
            print(f"  - {m_rep['icon']} {m_rep['name']:<20} : {m_rep['verdict']:<25} ({m_rep['confidence']:.1f}%) [{m_rep['latency_ms']} ms]")

        print("\n[Consensus Probability Distribution]")
        for k, v in c["probabilities"].items():
            print(f"  - {k}: {v * 100:.2f}%")

        print("\n[Inter-Modality Discrepancy Matrix]")
        for k, v in result["discrepancy_matrix"].items():
            print(f"  - {k}: {v}")
        print("="*65 + "\n")
        return

    model_name = getattr(args, "model", "mobilenet_v3")
    print(f"[*] Analyzing Media Payload: {args.video_path} [Model: {model_name.upper()}]")
    if args.claim:
        print(f"[*] Accompanying Claim / Caption: \"{args.claim}\"")
    if args.degrade:
        print("[*] Pre-processing with WhatsApp H.264 CRF 38 degradation simulator...")

    engine = VeriMediaEngine(model_name=model_name)
    result = engine.verify_media(
        video_path=args.video_path,
        claim_text=args.claim,
        simulate_whatsapp_degradation=args.degrade
    )

    print("\n" + "="*65)
    print("VERIMEDIA FORENSIC VERIFICATION REPORT")
    print("="*65)
    print(f"VERDICT: {result['verdict'].upper()} (Confidence: {result['confidence']}%)")
    print("\n[Probability Distribution]")
    for k, v in result["probabilities"].items():
        print(f"  - {k}: {v * 100:.2f}%")

    print("\n[Modality Forensic Breakdown]")
    for k, v in result["modality_scores"].items():
        print(f"  - {k}: {v}")

    print("\n[Inter-Modality Discrepancy Matrix]")
    for k, v in result["discrepancy_matrix"].items():
        print(f"  - {k}: {v}")

    print("\n[Performance Timings]")
    for k, v in result["performance_timings"].items():
        print(f"  - {k}: {v}")
    print("="*65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        prog="verimedia",
        description="VeriMedia — Edge-Optimized Multimodal Deepfake and Regional Misinformation Detection Engine"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Scan dataset and build manifests")
    p_inspect.add_argument("--dataset_path", type=str, default=str(config.dataset_root), help="Path to FaceForensics++ root")

    # degrade
    p_degrade = subparsers.add_parser("degrade", help="Simulate WhatsApp compression on video")
    p_degrade.add_argument("video_path", type=str, help="Path to video file")
    p_degrade.add_argument("--crf", type=int, default=38, help="H.264 CRF (default: 38)")
    p_degrade.add_argument("--fps", type=int, default=15, help="Target FPS (default: 15)")

    # quantize
    p_quantize = subparsers.add_parser("quantize", help="Export to ONNX and quantize to INT8")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Benchmark against System Verification Matrix")
    p_bench.add_argument("--iterations", type=int, default=50, help="Benchmark iteration count")

    # predict
    p_predict = subparsers.add_parser("predict", help="Verify a video and optional claim")
    p_predict.add_argument("video_path", type=str, help="Path to video file")
    p_predict.add_argument("--claim", type=str, default=None, help="Accompanying regional claim / caption text")
    p_predict.add_argument("--degrade", action="store_true", help="Simulate WhatsApp degradation before verification")
    p_predict.add_argument("--committee", action="store_true", help="Run multi-model forensic committee ensemble (Option A)")
    p_predict.add_argument(
        "--model",
        type=str,
        default="mobilenet_v3",
        choices=["mobilenet_v3", "xception", "efficientnet_v2", "resnet50", "mesonet", "temporal_lstm"],
        help="Model backbone architecture (when not using --committee)"
    )

    args = parser.parse_args()
    if args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "degrade":
        cmd_degrade(args)
    elif args.command == "quantize":
        cmd_quantize(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
