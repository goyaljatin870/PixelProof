"""
frame_cache.py - Multi-threaded lossless face-crop tensor pre-extraction for FaceForensics++.
Extracts N=8 evenly spaced 224x224 RGB uint8 facial crops per video into numpy .npy files
so multi-epoch GPU training runs without per-epoch OpenCV/CPU decoding bottlenecks.
"""

import os
import sys
import json
import time
import hashlib
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from src.config import config

_thread_local = threading.local()


def _get_cascade() -> Optional[cv2.CascadeClassifier]:
    if not hasattr(_thread_local, "cascade"):
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                _thread_local.cascade = cv2.CascadeClassifier(cascade_path)
            else:
                _thread_local.cascade = None
        except Exception:
            _thread_local.cascade = None
    return _thread_local.cascade


def cache_key_for_row(row: Dict) -> str:
    rel = str(row.get("relative_path") or row.get("video_path") or "")
    cat = str(row.get("category") or "clip")
    stem = Path(str(row.get("filename") or rel)).stem
    digest = hashlib.md5(rel.replace("\\", "/").encode("utf-8")).hexdigest()[:10]
    safe_cat = "".join(c if c.isalnum() or c in "-_" else "_" for c in cat)
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return f"{safe_cat}__{safe_stem}__{digest}.npy"


def extract_uniform_face_crops(
    video_path: str,
    num_frames: int = 8,
    crop_size: int = 224,
    span_sec: float = 4.5,
) -> np.ndarray:
    """
    Single-pass sequential capture extracting `num_frames` RGB crops of shape (num_frames, 224, 224, 3) uint8.
    Uses thread-local 320px Haar detection with temporal bbox carry-over fallback.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((num_frames, crop_size, crop_size, 3), dtype=np.uint8)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    if total_frames <= 0:
        span_limit = max(num_frames * 4, int(round(fps * span_sec)))
    else:
        span_limit = min(total_frames, max(num_frames, int(round(fps * span_sec))))

    targets = set(np.linspace(0, max(0, span_limit - 1), num_frames, dtype=int).tolist())
    max_idx = max(targets) if targets else 0

    cascade = _get_cascade()
    crops = []
    last_box: Optional[Tuple[int, int, int, int]] = None

    for idx in range(max_idx + 1):
        if not cap.grab():
            break
        if idx not in targets:
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            continue

        h, w, _ = frame.shape
        if cascade is not None:
            scale = 320.0 / max(w, 1)
            small_h = max(1, int(round(h * scale)))
            small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, small_h))
            try:
                faces = cascade.detectMultiScale(
                    small_gray, scaleFactor=1.15, minNeighbors=4, minSize=(20, 20)
                )
            except Exception:
                faces = ()
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                fx = int(fx / scale)
                fy = int(fy / scale)
                fw = int(fw / scale)
                fh = int(fh / scale)
                mx, my = int(0.2 * fw), int(0.2 * fh)
                last_box = (
                    max(0, fx - mx),
                    max(0, fy - my),
                    min(w, fx + fw + mx),
                    min(h, fy + fh + my),
                )

        if last_box is not None:
            x1, y1, x2, y2 = last_box
            crop = frame[y1:y2, x1:x2]
        else:
            m = min(h, w)
            cy, cx = h // 2, w // 2
            crop = frame[cy - m // 2 : cy + m // 2, cx - m // 2 : cx + m // 2]

        if crop.size == 0:
            crop = frame

        crop_rgb = cv2.cvtColor(cv2.resize(crop, (crop_size, crop_size)), cv2.COLOR_BGR2RGB)
        crops.append(crop_rgb)

    cap.release()

    if not crops:
        return np.zeros((num_frames, crop_size, crop_size, 3), dtype=np.uint8)

    while len(crops) < num_frames:
        crops.append(crops[-1].copy())

    return np.stack(crops[:num_frames], axis=0).astype(np.uint8)


def build_frame_cache(
    manifest_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    num_frames: int = 8,
    max_workers: int = 12,
) -> Dict[str, str]:
    manifest_file = Path(manifest_path) if manifest_path else config.manifest_dir / "manifest.csv"
    out_dir = Path(cache_dir) if cache_dir else config.project_root / "data" / "frame_cache_8f"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_file)
    rows = df.to_dict(orient="records")

    mapping: Dict[str, str] = {}
    todo = []
    expected_bytes = num_frames * 224 * 224 * 3

    for r in rows:
        vp = str(r["video_path"])
        key_name = cache_key_for_row(r)
        out_path = out_dir / key_name
        mapping[vp] = str(out_path)
        if not (out_path.exists() and out_path.stat().st_size >= expected_bytes):
            todo.append((vp, out_path))

    total_n = len(rows)
    print(f"[FrameCache] Total manifest videos: {total_n} | Already cached: {total_n - len(todo)} | To extract: {len(todo)}", flush=True)

    t0 = time.perf_counter()
    done_cnt = 0
    if todo:
        def _worker(item):
            vp, out_p = item
            arr = extract_uniform_face_crops(vp, num_frames=num_frames)
            tmp_p = out_p.with_suffix(".tmp.npy")
            np.save(str(tmp_p), arr)
            os.replace(str(tmp_p), str(out_p))
            return 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for _ in pool.map(_worker, todo):
                done_cnt += 1
                if done_cnt % 500 == 0 or done_cnt == len(todo):
                    dt = time.perf_counter() - t0
                    rate = done_cnt / max(dt, 1e-6)
                    rem = (len(todo) - done_cnt) / max(rate, 1e-6)
                    print(f"[FrameCache] Extracted {done_cnt}/{len(todo)} ({rate:.1f} vid/s, elapsed {dt/60:.1f}m, ETA {rem/60:.1f}m)", flush=True)

    index_path = out_dir / "cache_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f)

    total_dt = time.perf_counter() - t0
    print(f"[FrameCache] Completed in {total_dt/60:.2f} min. Index saved to {index_path}", flush=True)
    return mapping


if __name__ == "__main__":
    build_frame_cache()
