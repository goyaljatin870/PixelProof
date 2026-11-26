"""
config.py - VeriMedia Configuration Management
Defines dataset paths, edge constraints, degradation parameters, and model hyperparameters.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env if present
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()


@dataclass
class EdgeConstraints:
    """Target performance criteria for edge mobile deployment (Snapdragon 680 / 2GB RAM)."""
    max_latency_ms: float = float(os.getenv("EDGE_MAX_LATENCY_MS", 300.0))
    max_model_size_mb: float = float(os.getenv("EDGE_MAX_MODEL_SIZE_MB", 45.0))
    max_peak_ram_mb: float = float(os.getenv("EDGE_MAX_PEAK_RAM_MB", 180.0))
    target_auc: float = 0.895


@dataclass
class DegradationProfile:
    """WhatsApp & low-bandwidth simulated degradation parameters (H.264 + Opus)."""
    max_width: int = int(os.getenv("DEGRADE_MAX_WIDTH", 640))
    max_height: int = int(os.getenv("DEGRADE_MAX_HEIGHT", 480))
    target_fps: int = int(os.getenv("DEGRADE_TARGET_FPS", 15))
    crf: int = int(os.getenv("DEGRADE_CRF", 38))
    audio_bitrate: str = os.getenv("DEGRADE_AUDIO_BITRATE", "16k")
    audio_sample_rate: int = int(os.getenv("DEGRADE_AUDIO_SAMPLERATE", 16000))
    video_codec: str = "libx264"
    audio_codec: str = "libopus"
    fallback_audio_codec: str = "aac"


@dataclass
class ModelHyperparameters:
    """Hyperparameters for visual, audio, nlp branches, and late fusion MLP."""
    # Class labels: 0: Authentic, 1: Manipulated Payload, 2: Contextual Misinformation
    num_classes: int = 3
    class_names: list = field(default_factory=lambda: [
        "Authentic",
        "Manipulated Payload",
        "Contextual Misinformation"
    ])
    
    # Feature extraction dimensions
    visual_feature_dim: int = 2048      # EfficientNetV2 bottleneck
    student_visual_dim: int = 576       # MobileNetV3 bottleneck
    audio_feature_dim: int = 768        # Wav2Vec 2.0 / Acoustic Vocoder representation
    text_feature_dim: int = 768         # XLM-RoBERTa / Intent Claim embedding
    
    # Common projection dimension for cross-modality discrepancy matrix
    shared_projection_dim: int = 256
    
    # Fusion head structure
    fusion_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.3
    
    # Video sampling rate
    sample_fps: int = 5
    max_frames_per_video: int = 32
    
    # Distillation parameters
    distillation_alpha: float = 0.5     # Weight of KD loss vs cross-entropy
    distillation_temperature: float = 2.0


def _resolve_default_dataset_root() -> Path:
    """Resolves dataset root from env, local FF++ directory, or bundled sample dataset."""
    env_dir = os.getenv("FFPP_ROOT_DIR")
    if env_dir:
        return Path(env_dir)
    legacy_dir = Path(r"D:\ML\FaceForensics++_C23")
    if legacy_dir.exists():
        return legacy_dir
    return Path(__file__).resolve().parent.parent / "data" / "sample_dataset" / "FaceForensics++"


def _resolve_default_splits_dir(dataset_root: Path) -> Path:
    """Resolves official split directory matching the active dataset root."""
    env_dir = os.getenv("SPLITS_DIR")
    if env_dir:
        return Path(env_dir)
    if (dataset_root / "csv").exists():
        return dataset_root / "csv"
    return dataset_root / "splits"


@dataclass
class VeriMediaConfig:
    """Central VeriMedia Project Configuration."""
    # Root paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    
    # Dataset path (env override -> local FF++ corpus -> bundled sample dataset)
    dataset_root: Path = field(default_factory=_resolve_default_dataset_root)
    splits_csv_dir: Path = field(default_factory=lambda: _resolve_default_splits_dir(_resolve_default_dataset_root()))
    
    # Pipeline directories
    manifest_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "manifests")
    processed_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "processed")
    degraded_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "degraded_whatsapp")
    checkpoints_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "artifacts" / "checkpoints")
    reports_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "artifacts" / "reports")
    onnx_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "artifacts" / "onnx")
    
    # Execution settings
    device: str = os.getenv("DEVICE", "cpu")
    batch_size: int = int(os.getenv("BATCH_SIZE", 16))
    num_workers: int = int(os.getenv("NUM_WORKERS", 0))
    learning_rate: float = 1e-4
    
    # Sub-configurations
    edge: EdgeConstraints = field(default_factory=EdgeConstraints)
    degradation: DegradationProfile = field(default_factory=DegradationProfile)
    model: ModelHyperparameters = field(default_factory=ModelHyperparameters)

    def ensure_directories(self) -> None:
        """Create required artifact and data directories if they don't exist."""
        for p in [self.manifest_dir, self.processed_dir, self.degraded_dir,
                 self.checkpoints_dir, self.reports_dir, self.onnx_dir]:
            p.mkdir(parents=True, exist_ok=True)


# Default global instance
config = VeriMediaConfig()
