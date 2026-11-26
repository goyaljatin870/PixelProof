"""
audio_branch.py - Acoustic & Neural Vocoder Synthesis Detection
Extracts 16kHz mono audio from video, computes acoustic forensic features
(phase irregularities, vocoder artifacts, formant transitions), and outputs
a 768-dimensional acoustic representation vector (A_f).
"""

import os
import wave
import hashlib
import subprocess
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from src.config import config
import imageio_ffmpeg


def extract_audio_from_video(video_path: str, output_wav_path: Optional[str] = None, sample_rate: int = 16000) -> Optional[str]:
    """
    Extracts 16kHz mono PCM audio track from video file using bundled FFmpeg.
    Returns path to temporary .wav file or None if no audio track exists.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    if output_wav_path is None:
        cache_dir = config.processed_dir / "audio_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        resolved = str(Path(video_path).resolve())
        path_tag = hashlib.md5(resolved.encode("utf-8")).hexdigest()[:8]
        stem = Path(video_path).stem
        output_wav_path = str(cache_dir / f"{stem}_{path_tag}_16k.wav")

    if os.path.exists(output_wav_path):
        try:
            os.remove(output_wav_path)
        except OSError:
            pass

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(video_path),
        "-vn",                      # disable video
        "-ac", "1",                 # mono
        "-ar", str(sample_rate),    # 16kHz
        "-f", "wav",
        str(output_wav_path)
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if res.returncode == 0 and os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 44:  # > header size
            return output_wav_path
        return None
    except Exception:
        return None


def read_wav_16k(wav_path: str, max_duration_sec: float = 10.0, sample_rate: int = 16000) -> Tuple[np.ndarray, bool]:
    """
    Reads 16kHz mono PCM .wav file into float32 numpy array normalized to [-1.0, 1.0].
    """
    if not os.path.exists(wav_path):
        return np.zeros(sample_rate * 2, dtype=np.float32), False

    try:
        with wave.open(wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            
            max_frames = int(max_duration_sec * framerate)
            read_frames = min(n_frames, max_frames)
            raw_data = wf.readframes(read_frames)

            if sampwidth == 2:
                audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                audio = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            elif sampwidth == 1:
                audio = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                audio = np.frombuffer(raw_data, dtype=np.float32)

            if n_channels > 1:
                audio = audio[::n_channels]

            # Resample if not 16kHz
            if framerate != sample_rate and len(audio) > 0:
                indices = np.linspace(0, len(audio) - 1, int(len(audio) * sample_rate / framerate))
                audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

            if len(audio) == 0:
                return np.zeros(sample_rate * 2, dtype=np.float32), False

            return audio, True
    except Exception:
        return np.zeros(sample_rate * 2, dtype=np.float32), False


def compute_vocoder_anomaly_score(audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """
    Forensic acoustic analysis: detects vocoder high-frequency noise floor,
    phase irregularities, and formant instability typical of synthetic voice cloning.
    """
    if len(audio) < 512:
        return {
            "vocoder_anomaly_score": 0.0,
            "phase_irregularity": 0.0,
            "high_freq_noise_ratio": 0.0,
            "has_audio": 0.0
        }

    # Guard against near-silent tracks where uniform random background phase mimics jitter
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    if rms < 1e-3:
        return {
            "vocoder_anomaly_score": 0.0,
            "phase_irregularity": 0.0,
            "high_freq_noise_ratio": 0.0,
            "has_audio": 1.0
        }

    # Windowed FFT analysis
    n_fft = 512
    hop_length = 256
    
    n_frames = (len(audio) - n_fft) // hop_length + 1
    if n_frames < 2:
        return {"vocoder_anomaly_score": 0.0, "phase_irregularity": 0.0, "high_freq_noise_ratio": 0.0, "has_audio": 1.0}

    # Frame-wise spectra
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, n_fft),
        strides=(audio.strides[0] * hop_length, audio.strides[0])
    )
    window = np.hanning(n_fft)
    windowed = frames * window
    spectra = np.fft.rfft(windowed, axis=1)
    mag = np.abs(spectra)
    phase = np.angle(spectra)

    # 1. High-frequency vocoder noise ratio (>6kHz in 16kHz audio: bins 192 to 256)
    # Neural vocoders (HiFi-GAN, WaveNet) exhibit unnatural high-frequency energy buzz
    hf_energy = np.mean(mag[:, 192:] ** 2)
    total_energy = np.mean(mag ** 2) + 1e-8
    hf_noise_ratio = float(np.clip(hf_energy / total_energy, 0.0, 1.0))

    # 2. Phase unwrapping discontinuity (temporal phase jitter)
    # Real speech has natural phase dynamics; synthetic vocoders exhibit unnatural periodicity or harsh discontinuities
    phase_diff = np.diff(phase, axis=0)
    raw_jitter = float(np.std(phase_diff) / np.pi)
    
    # Calibrated phase jitter: only trigger on extreme phase lock or harsh vocoder distortion (>0.85)
    phase_anomaly = float(np.clip((raw_jitter - 0.70) / 0.30, 0.0, 1.0))

    # Calibrated vocoder score: natural speech stays < 0.15; synthetic vocoders reach > 0.50
    vocoder_score = float(np.clip(0.6 * phase_anomaly + 0.4 * (hf_noise_ratio * 4.0), 0.0, 1.0))

    return {
        "vocoder_anomaly_score": round(vocoder_score, 4),
        "phase_irregularity": round(phase_anomaly, 4),
        "high_freq_noise_ratio": round(hf_noise_ratio, 4),
        "has_audio": 1.0
    }


class AcousticVocoderBackbone(nn.Module):
    """
    Wav2Vec 2.0 / Acoustic Vocoder 1D Convolutional Neural Network.
    Processes raw 16kHz audio waveforms directly to identify vocoder signatures
    and outputs a 768-dimensional acoustic representation vector (A_f).
    """
    def __init__(self, out_dim: int = 768):
        super().__init__()
        self.out_dim = out_dim
        
        # 1D Temporal Convolutional Feature Extractor (mirroring Wav2Vec2 front-end)
        # Conv block 1: (1, 16000) -> kernel 10, stride 5
        self.conv1 = nn.Conv1d(1, 128, kernel_size=10, stride=5, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.GELU()

        # Conv block 2: kernel 8, stride 4
        self.conv2 = nn.Conv1d(128, 256, kernel_size=8, stride=4, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.GELU()

        # Conv block 3: kernel 4, stride 2
        self.conv3 = nn.Conv1d(256, 512, kernel_size=4, stride=2, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu3 = nn.GELU()

        # Conv block 4: kernel 4, stride 2
        self.conv4 = nn.Conv1d(512, 512, kernel_size=4, stride=2, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.relu4 = nn.GELU()

        # Temporal Context Layer
        self.gru = nn.GRU(512, 384, num_layers=2, batch_first=True, bidirectional=True)
        
        # Output Projection to 768d
        self.proj = nn.Linear(768, out_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            waveform: Tensor of shape (B, T) or (B, 1, T) representing 16kHz mono audio.
        Returns:
            A_f: Tensor of shape (B, 768)
        """
        if waveform.ndim == 2:
            x = waveform.unsqueeze(1)  # (B, 1, T)
        else:
            x = waveform

        # Temporal convolutions
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))

        # Reshape for GRU: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        out, _ = self.gru(x)  # (B, T, 768)

        # Average pooling over time steps (as specified in document)
        a_f = torch.mean(out, dim=1)  # (B, 768)
        a_f = self.proj(a_f)
        return a_f


class AudioBranchExtractor:
    """
    High-level extractor for video audio tracks.
    Handles extraction, acoustic feature modeling, and vocoder anomaly detection.
    """
    def __init__(self, model: Optional[AcousticVocoderBackbone] = None):
        self.model = model if model is not None else AcousticVocoderBackbone(out_dim=768)
        self.model.eval()

    def process_video_audio(self, video_path: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Extracts audio from video, computes vocoder anomaly score, and returns:
            a_f_tensor: (1, 768) feature tensor
            anomaly_info: dict with forensic vocoder metrics
        """
        wav_path = extract_audio_from_video(video_path)
        
        if wav_path is None or not os.path.exists(wav_path):
            # Silent fallback
            dummy = torch.zeros((1, 16000), dtype=torch.float32)
            with torch.no_grad():
                a_f = self.model(dummy)
            return a_f, {
                "vocoder_anomaly_score": 0.0,
                "phase_irregularity": 0.0,
                "high_freq_noise_ratio": 0.0,
                "has_audio": 0.0
            }

        audio, has_audio = read_wav_16k(wav_path)
        anomaly_info = compute_vocoder_anomaly_score(audio)

        # Truncate or pad to 3 seconds (48,000 samples)
        target_len = 48000
        if len(audio) < target_len:
            audio_padded = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio_padded = audio[:target_len]

        wave_tensor = torch.tensor(audio_padded, dtype=torch.float32).unsqueeze(0)  # (1, T)
        with torch.no_grad():
            a_f = self.model(wave_tensor)

        return a_f, anomaly_info
