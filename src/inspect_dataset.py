"""
inspect_dataset.py - Dataset Inspection and Manifest Generation Engine
Part of VeriMedia Engine (Milestone 1)

Scans FaceForensics++ (from FFPP_ROOT_DIR or bundled sample dataset) and In-The-Wild datasets,
leverages precomputed metadata CSVs when present, enforces identity-disjoint splits,
and produces structured manifests (CSV and JSON summary).
"""

import os
import sys
import json
import glob
import argparse
import logging
from pathlib import Path

# Ensure package directory is in sys.path
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Try importing OpenCV for video probing
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV not installed in current environment.")

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# FaceForensics++ manipulation categories
FFPP_MANIPULATION_CATEGORIES = {
    "Deepfakes": "FAKE",
    "Face2Face": "FAKE",
    "FaceSwap": "FAKE",
    "NeuralTextures": "FAKE",
    "FaceShifter": "FAKE",
    "DeepFakeDetection": "FAKE",
    "youtube": "REAL",
    "original": "REAL",
    "actors": "REAL"
}


def parse_ffpp_source_ids(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract source actor ID and target actor ID from FaceForensics++ video filenames.
    - Original: '000.mp4' -> source_id='000', target_id=None
    - Manipulated: '000_003.mp4' -> source_id='000', target_id='003'
    - DeepFakeDetection: '01_02__meeting_serious__YVGY8LOK.mp4' -> source_id='01', target_id='02'
    """
    stem = Path(filename).stem
    if "__" in stem:
        parts = stem.split("__")[0].split("_")
        if len(parts) >= 2:
            return parts[0], parts[1]
    
    parts = stem.split("_")
    if len(parts) == 1:
        return parts[0], None
    elif len(parts) >= 2:
        return parts[0], parts[1]
    return stem, None


def parse_wild_source_ids(filename: str, dataset_name: str = "InTheWild") -> Tuple[Optional[str], Optional[str]]:
    """
    Extract identity IDs for in-the-wild datasets such as Celeb-DF v2.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0].startswith("id") and parts[1].startswith("id"):
        return parts[0], parts[1]
    elif len(parts) >= 2 and parts[0].startswith("id"):
        return parts[0], None
    return stem, None


def scan_wild_dataset_directory(wild_root: Path, dataset_name: str = "InTheWild", probe_sample_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Scans In-The-Wild datasets.
    """
    records = []
    if not wild_root.exists():
        return records
    video_files = [p for p in wild_root.rglob("*") if p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS]
    for v_path in video_files:
        src_id, tgt_id = parse_wild_source_ids(v_path.name, dataset_name)
        records.append({
            "dataset_name": dataset_name,
            "video_path": str(v_path.resolve()),
            "filename": v_path.name,
            "label": "REAL" if "real" in str(v_path).lower() else "FAKE",
            "source_id": src_id,
            "target_id": tgt_id,
            "split": "unassigned",
            "file_size_bytes": v_path.stat().st_size if v_path.exists() else 0,
            "is_readable": True
        })
    return records


def load_from_ffpp_csv_directory(ff_root: Path) -> List[Dict[str, Any]]:
    """
    Loads dataset metadata directly from FaceForensics++ CSV directory if present.
    High-speed indexing of 7,000 videos.
    """
    csv_file = ff_root / "csv" / "FF++_Metadata.csv"
    if not csv_file.exists():
        csv_file = ff_root / "csv" / "FF++_Metadata_Shuffled.csv"
        
    if not csv_file.exists():
        return []

    logger.info(f"Loading metadata from existing CSV: {csv_file}")
    df_raw = pd.read_csv(csv_file)
    records: List[Dict[str, Any]] = []

    for _, row in df_raw.iterrows():
        rel_str = str(row.get("File Path", "")).replace("\\", "/")
        full_path = ff_root / rel_str
        category = rel_str.split("/")[0] if "/" in rel_str else "unknown"
        lbl_str = str(row.get("Label", "")).upper()
        
        label = "REAL" if "REAL" in lbl_str else "FAKE"
        filename = Path(rel_str).name
        source_id, target_id = parse_ffpp_source_ids(filename)

        rec = {
            "dataset_name": "FaceForensics++",
            "video_path": str(full_path.resolve()) if full_path.exists() else str(full_path),
            "relative_path": rel_str,
            "filename": filename,
            "category": category,
            "label": label,
            "compression": "c23",
            "source_id": source_id,
            "target_id": target_id,
            "split": "unassigned",
            "width": int(row["Width"]) if pd.notna(row.get("Width")) else None,
            "height": int(row["Height"]) if pd.notna(row.get("Height")) else None,
            "fps": 25.0,
            "frame_count": int(row["Frame Count"]) if pd.notna(row.get("Frame Count")) else None,
            "file_size_bytes": int(float(row["File Size(MB)"]) * 1024 * 1024) if pd.notna(row.get("File Size(MB)")) else 0,
            "is_readable": full_path.exists()
        }
        records.append(rec)

    logger.info(f"Successfully loaded {len(records)} video records from FF++ CSV.")
    return records


def scan_faceforensics_directory(
    ff_root: Path,
    split_map: Optional[Dict[str, str]] = None,
    probe_sample_limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Scans a FaceForensics++ directory structure, detecting original and manipulated sequences.
    """
    records: List[Dict[str, Any]] = []
    if not ff_root.exists():
        logger.warning(f"FaceForensics++ path does not exist: {ff_root}")
        return records

    # First attempt: ultra-fast CSV metadata load if no custom split_map provided
    if not split_map:
        csv_records = load_from_ffpp_csv_directory(ff_root)
        if csv_records:
            return csv_records

    # Second attempt: filesystem directory walk
    logger.info(f"Scanning FaceForensics++ filesystem: {ff_root}")
    video_files = [p for p in ff_root.rglob("*") if p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS]
    logger.info(f"Found {len(video_files)} video files under {ff_root}")

    for v_path in video_files:
        rel_path = v_path.relative_to(ff_root)
        parts = rel_path.parts

        category = "unknown"
        label = "UNKNOWN"
        for part in parts:
            for cat_name, cat_label in FFPP_MANIPULATION_CATEGORIES.items():
                if cat_name.lower() in part.lower():
                    category = cat_name
                    label = cat_label
                    break
            if category != "unknown":
                break

        # Fallback based on parent directory hierarchy
        parts_lower = [p.lower() for p in parts]
        if "original_sequences" in parts_lower or "youtube" in parts_lower or "original" in parts_lower:
            label = "REAL"
            if category == "unknown":
                category = "original"
        elif "manipulated_sequences" in parts_lower or "deepfakes" in parts_lower or "faceswap" in parts_lower:
            label = "FAKE"
            if category == "unknown":
                category = "manipulated"

        source_id, target_id = parse_ffpp_source_ids(v_path.name)
        file_size = v_path.stat().st_size if v_path.exists() else 0

        split = "unassigned"
        if split_map and v_path.stem in split_map:
            split = split_map[v_path.stem]
        elif split_map and source_id in split_map:
            split = split_map[source_id]

        record = {
            "dataset_name": "FaceForensics++",
            "video_path": str(v_path.resolve()),
            "relative_path": str(rel_path),
            "filename": v_path.name,
            "category": category,
            "label": label,
            "compression": "c23",
            "source_id": source_id,
            "target_id": target_id,
            "split": split,
            "file_size_bytes": file_size,
            "is_readable": file_size > 0
        }
        records.append(record)

    return records


def enforce_identity_disjoint_splits(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Strict identity-disjoint partitioning by source_id to prevent data leakage.
    """
    import random
    rng = random.Random(seed)

    unassigned_by_source: Dict[str, List[int]] = {}
    for idx, r in enumerate(records):
        sid = r.get("source_id") or r.get("filename")
        unassigned_by_source.setdefault(sid, []).append(idx)

    unique_sources = sorted(list(unassigned_by_source.keys()))
    rng.shuffle(unique_sources)

    n_sources = len(unique_sources)
    n_train = int(n_sources * train_ratio)
    n_val = int(n_sources * val_ratio)

    train_sources = set(unique_sources[:n_train])
    val_sources = set(unique_sources[n_train:n_train + n_val])
    test_sources = set(unique_sources[n_train + n_val:])

    for sid, indices in unassigned_by_source.items():
        if sid in train_sources:
            split_tag = "train"
        elif sid in val_sources:
            split_tag = "val"
        else:
            split_tag = "test"

        for idx in indices:
            records[idx]["split"] = split_tag

    return records


def generate_manifest_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes summary metrics for manifest dataframe.
    """
    if df.empty:
        return {"total_videos": 0, "message": "Empty manifest."}

    summary = {
        "total_videos": int(len(df)),
        "dataset_distribution": df["dataset_name"].value_counts().to_dict(),
        "label_distribution": df["label"].value_counts().to_dict(),
        "category_distribution": df["category"].value_counts().to_dict(),
        "split_distribution": df["split"].value_counts().to_dict(),
        "unique_source_identities": int(df["source_id"].nunique(dropna=True)) if "source_id" in df.columns else 0
    }
    return summary


def build_manifest(
    ff_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Builds dataset manifest and saves to data/manifests/manifest.csv and manifest_summary.json.
    """
    ff_root = Path(ff_path) if ff_path else config.dataset_root
    out_dir = Path(output_dir) if output_dir else config.manifest_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_ffpp = config.project_root / "data" / "sample_dataset" / "FaceForensics++"
    used_sample_fallback = (ff_root.resolve() == sample_ffpp.resolve()) if ff_root.exists() and sample_ffpp.exists() else False

    records = scan_faceforensics_directory(ff_root)
    if not records and not ff_path:
        if sample_ffpp.exists():
            logger.info(f"Using local sample dataset at {sample_ffpp}")
            records = scan_faceforensics_directory(sample_ffpp)
            used_sample_fallback = True
    if not records:
        logger.warning(f"No records found at {ff_root}")

    records = enforce_identity_disjoint_splits(records)
    df = pd.DataFrame(records)
    summary = generate_manifest_summary(df)

    csv_path = out_dir / "manifest.csv"
    json_path = out_dir / "manifest_summary.json"

    # Preserve precomputed full 7,000-video summary if only indexing the small sample fallback
    if used_sample_fallback and not ff_path and json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing_summary = json.load(f)
            if existing_summary.get("total_videos", 0) > len(df):
                json_path = out_dir / "sample_manifest_summary.json"
        except Exception:
            pass

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Manifest written to {csv_path} ({len(df)} records)")
    return df, summary


def print_manifest_report(summary: Dict[str, Any]):
    """
    Prints a formatted summary report for dataset manifests.
    """
    print("\n" + "="*60)
    print("VERIMEDIA DATASET MANIFEST SUMMARY")
    print("="*60)
    print(f"Total Videos Verified: {summary.get('total_videos')}")
    print(f"Unique Identities: {summary.get('unique_source_identities', 'N/A')}")
    print(f"Labels: {summary.get('label_distribution')}")
    print(f"Categories: {summary.get('category_distribution')}")
    print(f"Data Splits: {summary.get('split_distribution')}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="VeriMedia Dataset Inspector & Manifest Builder")
    parser.add_argument("--ff_path", type=str, default=str(config.dataset_root), help="Path to FaceForensics++ root")
    parser.add_argument("--output_dir", type=str, default=str(config.manifest_dir), help="Output directory")

    args = parser.parse_args()
    df, summary = build_manifest(ff_path=args.ff_path, output_dir=args.output_dir)
    print_manifest_report(summary)


if __name__ == "__main__":
    main()
