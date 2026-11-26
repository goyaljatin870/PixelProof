"""
create_sample_dataset.py - Generates a minimal sample dataset structure
with dummy/synthetic MP4 video files for pipeline testing and verification
when full multi-gigabyte FaceForensics++ or Celeb-DF files are pending download.
"""

import sys
import json
from pathlib import Path
import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


def create_dummy_mp4(output_path: Path, width: int = 256, height: int = 256, fps: int = 15, num_frames: int = 30):
    """
    Creates a minimal playable MP4 video with moving color blocks.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if OPENCV_AVAILABLE:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Draw moving circle/rectangle to simulate dynamic visual input
            cx = int((width // 2) + 40 * np.sin(i * 0.2))
            cy = int((height // 2) + 40 * np.cos(i * 0.2))
            cv2.circle(frame, (cx, cy), 35, (0, 200, 255), -1)
            cv2.putText(frame, f"F:{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            out.write(frame)
        out.release()
    else:
        # Fallback binary placeholder if opencv is not yet installed
        output_path.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 2048)


def generate_sample_ffpp(base_dir: Path):
    """
    Builds FaceForensics++ mock directory structure with c23 and c40 videos and splits.
    """
    print(f"Generating mock FaceForensics++ dataset at: {base_dir}")
    splits_dir = base_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create official splits: train (actors 000..005), val (actor 006), test (actor 007)
    train_splits = [["000", "001"], ["002", "003"], ["004", "005"]]
    val_splits = [["006"]]
    test_splits = [["007"]]

    with open(splits_dir / "train.json", "w") as f:
        json.dump(train_splits, f, indent=2)
    with open(splits_dir / "val.json", "w") as f:
        json.dump(val_splits, f, indent=2)
    with open(splits_dir / "test.json", "w") as f:
        json.dump(test_splits, f, indent=2)

    # 2. Original sequences
    orig_c23 = base_dir / "original_sequences" / "youtube" / "c23" / "videos"
    for actor in ["000", "001", "002", "003", "004", "005", "006", "007"]:
        create_dummy_mp4(orig_c23 / f"{actor}.mp4")

    # 3. Manipulated sequences: Deepfakes, Face2Face across c23 & c40
    df_c23 = base_dir / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    df_c40 = base_dir / "manipulated_sequences" / "Deepfakes" / "c40" / "videos"
    f2f_c23 = base_dir / "manipulated_sequences" / "Face2Face" / "c23" / "videos"

    pairs = [("000", "001"), ("002", "003"), ("004", "005"), ("006", "007")]
    for s, t in pairs:
        create_dummy_mp4(df_c23 / f"{s}_{t}.mp4")
        create_dummy_mp4(df_c40 / f"{s}_{t}.mp4")
        create_dummy_mp4(f2f_c23 / f"{s}_{t}.mp4")

    print("Sample FaceForensics++ generated successfully.")


def generate_sample_wild(base_dir: Path):
    """
    Builds sample In-The-Wild (Celeb-DF v2 style) dataset directory.
    """
    print(f"Generating mock In-The-Wild dataset at: {base_dir}")
    real_dir = base_dir / "Celeb-real"
    synth_dir = base_dir / "Celeb-synthesis"
    yt_real_dir = base_dir / "YouTube-real"

    for i in range(4):
        create_dummy_mp4(real_dir / f"id{i}_0000.mp4")
        create_dummy_mp4(yt_real_dir / f"0000{i}.mp4")

    create_dummy_mp4(synth_dir / "id0_id1_0000.mp4")
    create_dummy_mp4(synth_dir / "id2_id3_0000.mp4")

    # Test list file
    test_file = base_dir / "List_of_testing_videos.txt"
    test_lines = [
        "1 Celeb-synthesis/id0_id1_0000.mp4\n",
        "0 Celeb-real/id0_0000.mp4\n"
    ]
    test_file.write_text("".join(test_lines))
    print("Sample In-The-Wild dataset generated successfully.")


if __name__ == "__main__":
    data_root = Path("data/sample_dataset")
    generate_sample_ffpp(data_root / "FaceForensics++")
    generate_sample_wild(data_root / "in_the_wild")
    print(f"\nSample datasets created under: {data_root.resolve()}")
