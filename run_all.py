"""
run_all.py - Master VS Code Execution Script for VeriMedia
Run this file directly in VS Code (Press F5 or click 'Run Python File' in the top right).
Provides an interactive menu and an automated end-to-end demonstration.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Ensure package root is in sys.path
_pkg_root = Path(__file__).resolve().parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from src.config import config
from src.inspect_dataset import build_manifest, print_manifest_report
from src.degradation_pipeline import degrade_video_whatsapp
from src.quantization import quantize_onnx_to_int8
from src.benchmark import benchmark_edge_model
from src.pipeline import VeriMediaEngine


def print_banner():
    print(rf"""
================================================================================
   ____   __           _  __  ___          ___         
  / __/  / /  ___ ____(_) / / / _ \___ ___ / _/__ ____ _
 _\ \   / _ \/ _ `/ __/ / _ \/ // / -_) -_) _/ _ `/ _ `/
/___/  /_//_/\_,_/_/ /_/_//_/____/\__/\__/_/ \_,_/\_, / 
                                                  /___/  
        VeriMedia: Multimodal Deepfake & Regional Misinformation Engine
================================================================================
Dataset Root: {config.dataset_root}
Edge Target:  Sub-300ms Latency | <= 45MB Quantized INT8 Payload
================================================================================
    """)


def run_full_end_to_end_demo():
    print("\n[STEP 1/5] Verifying Dataset Manifest...")
    manifest_csv = config.manifest_dir / "manifest.csv"
    if not manifest_csv.exists():
        df, summary = build_manifest()
        print_manifest_report(summary)
    else:
        print(f" Manifest already indexed at: {manifest_csv}")

    print("\n[STEP 2/5] Quantizing Distilled Student Model to 8-Bit ONNX (QUInt8)...")
    q_res = quantize_onnx_to_int8()
    print(f" FP32 Model Size: {q_res['fp32_size_mb']} MB")
    print(f" INT8 Model Size: {q_res['int8_size_mb']} MB (Target <= 45MB: {'PASS ✅' if q_res['is_size_compliant'] else 'FAIL ❌'})")
    print(f" Compression Ratio: {q_res['compression_ratio']}x reduction")

    print("\n[STEP 3/5] Benchmarking Against System Verification Matrix...")
    b_res = benchmark_edge_model(num_iterations=25)
    m = b_res["metrics"]
    print(f" Verification Status: {b_res['status']}")
    print(f" - Latency: {m['inference_latency']['mean_ms']} ms (Target < 300 ms: {'PASS ✅' if m['inference_latency']['compliant'] else 'FAIL ❌'})")
    print(f" - Model Size: {m['model_size']['value_mb']} MB (Target <= 45 MB: {'PASS ✅' if m['model_size']['compliant'] else 'FAIL ❌'})")
    print(f" - Active RAM: {m['active_memory_footprint']['active_delta_mb']} MB (Target < 180 MB: {'PASS ✅' if m['active_memory_footprint']['compliant'] else 'FAIL ❌'})")

    print("\n[STEP 4/5] Testing Forensic Inference on Real Video...")
    from src.pipeline import COMMITTEE_MODEL_METADATA
    all_pth_present = all(
        (config.checkpoints_dir / f"verimedia_{m}.pth").exists()
        for m in COMMITTEE_MODEL_METADATA.keys()
    )
    engine = VeriMediaEngine(model_name="committee", strict_trained_only=all_pth_present)
    sample_real = str(config.dataset_root / "original" / "006.mp4")
    if not os.path.exists(sample_real):
        sample_real = str(config.dataset_root / "original" / "000.mp4")
    if not os.path.exists(sample_real):
        fallback_real = _pkg_root / "data" / "sample_dataset" / "FaceForensics++" / "original_sequences" / "youtube" / "c23" / "videos" / "000.mp4"
        if fallback_real.exists():
            sample_real = str(fallback_real)
    if os.path.exists(sample_real):
        res_real = engine.verify_media(sample_real, claim_text="Official parliamentary briefing")
        print(f" Authentic Sample: {Path(sample_real).name}")
        print(f"   -> Verdict: {res_real['verdict'].upper()} (Confidence: {res_real['confidence']}%)")
        print(f"   -> SRM Residual Magnitude: {res_real['modality_scores']['visual_srm_residual_level']}")
        print(f"   -> Class Probabilities: {res_real['probabilities']}")
    else:
        print(f" Sample not found at: {sample_real}")

    print("\n[STEP 5/5] Testing Forensic Inference on Deepfake Video with Deceptive Claim + WhatsApp Compression...")
    sample_fake = str(config.dataset_root / "Deepfakes" / "006_002.mp4")
    if not os.path.exists(sample_fake):
        sample_fake = str(config.dataset_root / "Deepfakes" / "000_003.mp4")
    if not os.path.exists(sample_fake):
        fallback_fake = _pkg_root / "data" / "sample_dataset" / "FaceForensics++" / "manipulated_sequences" / "Deepfakes" / "c23" / "videos" / "000_001.mp4"
        if fallback_fake.exists():
            sample_fake = str(fallback_fake)
    if os.path.exists(sample_fake):
        res_fake = engine.verify_media(
            sample_fake,
            claim_text="BREAKING: Secret audio leaked before regional election! Must watch!!",
            simulate_whatsapp_degradation=True
        )
        print(f" Deepfake Sample: {Path(sample_fake).name}")
        print(f"   -> Verdict: {res_fake['verdict'].upper()} (Confidence: {res_fake['confidence']}%)")
        print(f"   -> Discrepancy Matrix: {res_fake['discrepancy_matrix']}")
        print(f"   -> Class Probabilities: {res_fake['probabilities']}")
        deg = res_fake["diagnostics"]["simulated_whatsapp_degradation"]
        if deg:
            print(f"   -> WhatsApp Degradation Applied: {deg['compression_ratio']}x compression reduction (CRF 38)")
    else:
        print(f" Sample not found at: {sample_fake}")

    print("\n" + "="*80)
    print("✅ End-to-End Demonstration Complete!")
    print("To launch the graphical dashboard, run: python -m streamlit run frontend/app.py")
    print("="*80 + "\n")


def launch_streamlit():
    print("\n[*] Launching Streamlit Web Dashboard in your default browser...")
    app_path = _pkg_root / "frontend" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


def run_tests():
    print("\n[*] Running pytest suite across all modules...")
    subprocess.run([sys.executable, "-m", "pytest", str(_pkg_root / "tests")])


def main():
    print_banner()

    # Check if arguments or interactive selection
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["--demo", "-d", "demo"]:
            run_full_end_to_end_demo()
            return
        elif arg in ["--dashboard", "-s", "dashboard", "streamlit"]:
            launch_streamlit()
            return
        elif arg in ["--test", "-t", "test", "tests"]:
            run_tests()
            return

    # Interactive Menu
    print("Please select an action to run in VS Code:")
    print("  [1] Run Full End-to-End Pipeline Demo (Recommended)")
    print("  [2] Launch Interactive Streamlit Web Dashboard")
    print("  [3] Run System Verification Matrix Benchmark")
    print("  [4] Run Full Unit & Integration Test Suite (PyTest)")
    print("  [5] Exit")
    
    choice = input("\nEnter choice [1-5] (default: 1): ").strip()
    if choice == "2":
        launch_streamlit()
    elif choice == "3":
        b_res = benchmark_edge_model()
        print("\nBenchmark Result:")
        print(b_res)
    elif choice == "4":
        run_tests()
    elif choice == "5":
        print("Exiting.")
    else:
        run_full_end_to_end_demo()


if __name__ == "__main__":
    main()
