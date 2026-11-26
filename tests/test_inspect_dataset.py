"""
Unit tests for inspect_dataset.py
Validates parsing logic, leakage prevention, and manifest creation.
"""

import os
import json
import tempfile
from pathlib import Path
import pytest
import pandas as pd

from src.inspect_dataset import (
    parse_ffpp_source_ids,
    parse_wild_source_ids,
    enforce_identity_disjoint_splits,
    generate_manifest_summary,
    scan_faceforensics_directory,
    scan_wild_dataset_directory
)


def test_parse_ffpp_source_ids():
    # Original video
    src, tgt = parse_ffpp_source_ids("042.mp4")
    assert src == "042"
    assert tgt is None

    # Manipulated video
    src, tgt = parse_ffpp_source_ids("042_189.mp4")
    assert src == "042"
    assert tgt == "189"


def test_parse_wild_source_ids():
    # Celeb-real
    src, tgt = parse_wild_source_ids("id0_0000.mp4")
    assert src == "id0"
    assert tgt is None

    # Celeb-synthesis
    src, tgt = parse_wild_source_ids("id0_id16_0000.mp4")
    assert src == "id0"
    assert tgt == "id16"


def test_identity_disjoint_splits_no_leakage():
    # Create sample records with 10 identities and multiple videos per identity
    sample_records = []
    for i in range(10):
        sid = f"actor_{i}"
        for j in range(4):
            sample_records.append({
                "video_path": f"/path/to/actor_{i}_clip_{j}.mp4",
                "filename": f"actor_{i}_clip_{j}.mp4",
                "source_id": sid,
                "split": "unassigned"
            })

    # Run partition
    partitioned = enforce_identity_disjoint_splits(sample_records, train_ratio=0.6, val_ratio=0.2, seed=42)

    # Verify no source ID appears in more than one split
    splits_per_source = {}
    for r in partitioned:
        sid = r["source_id"]
        split = r["split"]
        assert split in {"train", "val", "test"}
        if sid not in splits_per_source:
            splits_per_source[sid] = set()
        splits_per_source[sid].add(split)

    for sid, splits in splits_per_source.items():
        assert len(splits) == 1, f"Identity leakage detected! Source {sid} spans splits: {splits}"


def test_scan_directory_and_manifest_generation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Build mock FF++ directory structure
        c23_orig = tmp_path / "original_sequences" / "youtube" / "c23" / "videos"
        c23_orig.mkdir(parents=True)
        (c23_orig / "001.mp4").write_text("dummy video")
        (c23_orig / "002.mp4").write_text("dummy video")

        c23_fake = tmp_path / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
        c23_fake.mkdir(parents=True)
        (c23_fake / "001_002.mp4").write_text("dummy video")

        records = scan_faceforensics_directory(tmp_path, split_map={}, probe_sample_limit=None)
        assert len(records) == 3

        # Check labels
        df = pd.DataFrame(records)
        df_real = df[df["label"] == "REAL"]
        df_fake = df[df["label"] == "FAKE"]
        assert len(df_real) == 2
        assert len(df_fake) == 1

        # Check summary
        summary = generate_manifest_summary(df)
        assert summary["total_videos"] == 3
        assert summary["label_distribution"]["REAL"] == 2
        assert summary["label_distribution"]["FAKE"] == 1
