"""
degradation_pipeline.py - WhatsApp & Low-Bandwidth Synthetic Degradation Pipeline
Simulates aggressive messaging platform compression using FFmpeg:
480p downscaling, 15 FPS throttling, H.264 CRF 38, and 16kbps Opus/AAC audio transcoding.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm import tqdm

import imageio_ffmpeg
from src.config import config

logger = logging.getLogger(__name__)


def check_audio_stream_exists(video_path: str, ffmpeg_exe: str) -> bool:
    """
    Checks whether a video file contains at least one audio stream.
    """
    cmd = [ffmpeg_exe, "-i", str(video_path)]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        for line in res.stderr.splitlines():
            if "Audio:" in line:
                return True
        return False
    except Exception:
        return False


def degrade_video_whatsapp(
    input_path: str,
    output_path: Optional[str] = None,
    crf: Optional[int] = None,
    target_fps: Optional[int] = None,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None
) -> Dict[str, Any]:
    """
    Applies WhatsApp / dark-social degradation filter chain to an input video:
    - Downscaled to max 480p (640x480, preserving aspect ratio)
    - Throttled to 15 fps
    - H.264 CRF 38 (extreme quantization noise)
    - 16kHz mono Opus / AAC at 16kbps (if audio stream exists)
    
    Args:
        input_path: Path to pristine input video.
        output_path: Path to degraded output video. If None, saves in config.degraded_dir.
        crf: Override CRF (default from config: 38).
        target_fps: Override target fps (default: 15).
        max_width: Override max width (default: 640).
        max_height: Override max height (default: 480).
        
    Returns:
        Dict with degradation diagnostics (orig_size, degraded_size, compression_ratio, success, error).
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    crf = config.degradation.crf if crf is None else crf
    target_fps = config.degradation.target_fps if target_fps is None else target_fps
    max_width = config.degradation.max_width if max_width is None else max_width
    max_height = config.degradation.max_height if max_height is None else max_height

    if output_path is None:
        config.degraded_dir.mkdir(parents=True, exist_ok=True)
        stem = input_file.stem
        output_file = config.degraded_dir / f"{stem}_degraded_crf{crf}.mp4"
    else:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    has_audio = check_audio_stream_exists(str(input_file), ffmpeg_exe)

    # Video filter: scale to max 640:480 while preserving aspect ratio, throttled to 15 fps
    vf = f"scale={max_width}:{max_height}:force_original_aspect_ratio=decrease,fps={target_fps}"

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(input_file),
        "-vf", vf,
        "-c:v", config.degradation.video_codec,
        "-crf", str(crf),
        "-preset", "superfast",
        "-pix_fmt", "yuv420p"
    ]

    if has_audio:
        # Check if libopus is supported, otherwise fallback to aac
        cmd.extend([
            "-c:a", config.degradation.fallback_audio_codec,  # portable aac fallback for mp4 container
            "-b:a", config.degradation.audio_bitrate,
            "-ar", str(config.degradation.audio_sample_rate),
            "-ac", "1"
        ])
    else:
        cmd.append("-an")  # No audio stream in source

    cmd.append(str(output_file))

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if output_file.exists() and output_file.stat().st_size > 0:
            orig_size = input_file.stat().st_size
            degraded_size = output_file.stat().st_size
            comp_ratio = orig_size / max(1, degraded_size)

            return {
                "success": True,
                "input_path": str(input_file),
                "output_path": str(output_file),
                "orig_size_bytes": orig_size,
                "degraded_size_bytes": degraded_size,
                "compression_ratio": round(comp_ratio, 2),
                "crf": crf,
                "fps": target_fps,
                "had_audio": has_audio
            }
        else:
            return {
                "success": False,
                "input_path": str(input_file),
                "error": f"FFmpeg failed: {res.stderr[-500:] if res.stderr else 'Empty file generated'}"
            }
    except Exception as e:
        return {
            "success": False,
            "input_path": str(input_file),
            "error": str(e)
        }


def batch_degrade_dataset(
    video_paths: List[str],
    output_dir: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Batch executes the degradation pipeline over a list of videos.
    """
    out_dir = Path(output_dir) if output_dir else config.degraded_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if limit is not None:
        video_paths = video_paths[:limit]

    results = []
    logger.info(f"Starting WhatsApp degradation pipeline on {len(video_paths)} videos...")

    for v_path in tqdm(video_paths, desc="Simulating WhatsApp Compression"):
        res = degrade_video_whatsapp(v_path, output_path=None)
        results.append(res)

    successful = [r for r in results if r.get("success")]
    logger.info(f"Degradation pipeline complete: {len(successful)}/{len(video_paths)} videos processed successfully.")
    return results
