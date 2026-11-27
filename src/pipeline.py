"""
pipeline.py - VeriMedia Multimodal End-to-End Forensic Engine & Committee Ensemble
Orchestrates Visual (SRM), Audio (Vocoder), NLP (Claim Verification), and
Multi-Model Committee Consensus evaluation for inbound media payloads.
"""

import time
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from src.visual_branch import VideoFrameExtractor, StudentVisualBackbone
from src.audio_branch import AudioBranchExtractor, read_wav_16k, extract_audio_from_video, compute_vocoder_anomaly_score
from src.nlp_branch import NLPBranchExtractor, analyze_claim_sensationalism
from src.fusion_model import MultimodalLateFusionHead
from src.degradation_pipeline import degrade_video_whatsapp
from src.models_zoo import VeriMediaZooModel
from src.config import config


COMMITTEE_MODEL_METADATA = {
    "mobilenet_v3": {
        "name": "MobileNetV3-Edge",
        "role": "Lightweight Edge Baseline",
        "specialty": "Sub-millisecond spatial-frequency baseline & mobile efficiency",
        "icon": "⚡"
    },
    "xception": {
        "name": "XceptionNet",
        "role": "FaceForensics++ Canonical Benchmark",
        "specialty": "Facial reenactment boundaries & GAN blending seam detection",
        "icon": "🔬"
    },
    "mesonet": {
        "name": "MesoInception-4",
        "role": "Microscopic Noise Specialist",
        "specialty": "Steganalysis frequency noise & microscopic compression block artifacts",
        "icon": "🔍"
    },
    "temporal_lstm": {
        "name": "Temporal-LSTM",
        "role": "Temporal Motion Specialist",
        "specialty": "Multi-frame flickering, unnatural eye-blink periodicity & motion drift",
        "icon": "⏱️"
    },
    "resnet50": {
        "name": "ResNet-50",
        "role": "Deep Residual Network",
        "specialty": "Multi-scale deep residual feature extraction across facial anatomy",
        "icon": "🧠"
    },
    "efficientnet_v2": {
        "name": "EfficientNetV2",
        "role": "DFDC Competition Top Performer",
        "specialty": "Fused-MBConv spatial representations & subtle texture manipulation",
        "icon": "💎"
    }
}


class ForensicCommitteeEngine:
    """
    Autonomous Multi-Model Forensic Committee.
    Executes an ensemble of specialist deepfake architectures concurrently,
    aggregating individual model verdicts into a robust Master Consensus.
    Supports dynamic hot-reloading of custom trained .pth model checkpoints.
    """
    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        device: Optional[str] = None,
        strict_trained_only: bool = False,
    ):
        self.device = device or config.device
        self.strict_trained_only = bool(strict_trained_only)
        self.model_names = model_names or [
            "mobilenet_v3",
            "xception",
            "efficientnet_v2",
            "mesonet",
            "temporal_lstm",
            "resnet50"
        ]

        # Initialize shared extractors (aligned with 8-frame uniform ~4.5s training crops)
        self.frame_extractor = VideoFrameExtractor(
            target_fps=2,
            max_frames=8
        )
        self.nlp_extractor = NLPBranchExtractor()

        # Load committee models, per-model .pkl calibrators, and stacked ensemble .pkl
        self.models: Dict[str, VeriMediaZooModel] = {}
        self.checkpoint_statuses: Dict[str, Dict[str, Any]] = {}
        self.calibrators: Dict[str, Dict[str, Any]] = {}
        self.stacked_ensemble: Optional[Dict[str, Any]] = None
        self._init_committee()

    def _init_committee(self):
        """Initializes and loads trained .pth weights, .pkl calibrators, and ensemble .pkl."""
        for m_name in self.model_names:
            self._load_single_model(m_name)
        ens_path = config.checkpoints_dir / "verimedia_committee_ensemble.pkl"
        if ens_path.exists():
            try:
                import pickle
                with open(ens_path, "rb") as f:
                    self.stacked_ensemble = pickle.load(f)
            except Exception:
                self.stacked_ensemble = None

    def _load_single_model(self, m_name: str):
        """Instantiates a model architecture and loads its .pth checkpoint + _calibrator.pkl."""
        m_name = m_name.lower()
        model = VeriMediaZooModel(backbone_type=m_name).to(self.device)
        model.eval()

        ckpt_file = config.checkpoints_dir / f"verimedia_{m_name}.pth"
        pkl_file = config.checkpoints_dir / f"verimedia_{m_name}_calibrator.pkl"
        loaded = False
        is_trained = False
        batches_tracked = 0
        ckpt_size_mb = 0.0
        calibrator_loaded = False
        opt_thresh = 0.50

        if ckpt_file.exists():
            try:
                ckpt_size_mb = round(ckpt_file.stat().st_size / (1024 * 1024), 2)
                state = torch.load(ckpt_file, map_location=self.device)
                model.load_state_dict(state)
                loaded = True
                bt = getattr(model.fusion_head.bn1, "num_batches_tracked", None)
                batches_tracked = int(bt.item()) if bt is not None else 0
                is_trained = batches_tracked > 0
            except Exception:
                loaded = False
                is_trained = False

        if pkl_file.exists():
            try:
                import pickle
                with open(pkl_file, "rb") as f:
                    bundle = pickle.load(f)
                self.calibrators[m_name] = bundle
                calibrator_loaded = True
                opt_thresh = float(bundle.get("optimal_threshold", 0.50))
            except Exception:
                calibrator_loaded = False

        self.models[m_name] = model
        self.checkpoint_statuses[m_name] = {
            "checkpoint_path": str(ckpt_file),
            "calibrator_path": str(pkl_file),
            "exists": ckpt_file.exists(),
            "loaded": loaded,
            "is_trained": is_trained,
            "batches_tracked": batches_tracked,
            "calibrator_loaded": calibrator_loaded,
            "optimal_threshold": round(opt_thresh, 4),
            "size_mb": ckpt_size_mb
        }

    def reload_checkpoint(self, model_name: str, checkpoint_bytes_or_path: Any) -> Dict[str, Any]:
        """
        Dynamically hot-reloads a trained .pth checkpoint file for any specialist model.
        
        Args:
            model_name: Target model identifier (e.g. 'xception', 'mobilenet_v3')
            checkpoint_bytes_or_path: Bytes content or file path to new .pth file
            
        Returns:
            Status dictionary with loading outcome.
        """
        model_name = model_name.lower()
        if model_name not in COMMITTEE_MODEL_METADATA:
            return {"success": False, "error": f"Unknown model architecture: {model_name}"}

        dest_ckpt = config.checkpoints_dir / f"verimedia_{model_name}.pth"
        dest_ckpt.parent.mkdir(parents=True, exist_ok=True)

        try:
            if isinstance(checkpoint_bytes_or_path, (str, Path)):
                import shutil
                shutil.copyfile(str(checkpoint_bytes_or_path), str(dest_ckpt))
            elif hasattr(checkpoint_bytes_or_path, "read"):
                with open(dest_ckpt, "wb") as f:
                    f.write(checkpoint_bytes_or_path.read())
            elif isinstance(checkpoint_bytes_or_path, bytes):
                with open(dest_ckpt, "wb") as f:
                    f.write(checkpoint_bytes_or_path)

            # Reload model architecture in-memory with new weights
            self._load_single_model(model_name)
            status = self.checkpoint_statuses.get(model_name, {})

            if status.get("loaded"):
                return {
                    "success": True,
                    "model_name": model_name,
                    "size_mb": status.get("size_mb", 0.0),
                    "message": f"Successfully loaded trained weights into {COMMITTEE_MODEL_METADATA[model_name]['name']}!"
                }
            else:
                return {
                    "success": False,
                    "error": f"Checkpoint file saved but layer shapes failed validation for {model_name}."
                }
        except Exception as e:
            return {"success": False, "error": f"Failed to reload checkpoint: {str(e)}"}

    def get_committee_status(self) -> Dict[str, Any]:
        """Returns metadata and checkpoint readiness for all committee specialists."""
        status = {}
        for m_name in self.model_names:
            meta = COMMITTEE_MODEL_METADATA.get(m_name, {})
            ckpt_info = self.checkpoint_statuses.get(m_name, {})
            status[m_name] = {
                **meta,
                **ckpt_info
            }
        return status

    def has_full_trained_weights(self) -> bool:
        """Returns True when all specialist .pth weights, .pkl calibrators, and stacked ensemble .pkl are loaded."""
        if self.stacked_ensemble is None:
            return False
        return all(
            self.checkpoint_statuses.get(m, {}).get("is_trained", False)
            and self.checkpoint_statuses.get(m, {}).get("calibrator_loaded", False)
            for m in self.model_names
        )

    def verify_media_committee(
        self,
        video_path: str,
        claim_text: Optional[str] = None,
        simulate_whatsapp_degradation: bool = False,
        enabled_models: Optional[List[str]] = None,
        decision_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Runs comprehensive multi-model consensus verification across the committee.
        """
        t_start = time.perf_counter()
        active_video_path = video_path
        degradation_info = None

        # 0. Optional synthetic WhatsApp compression simulation
        if simulate_whatsapp_degradation:
            deg_res = degrade_video_whatsapp(video_path)
            if deg_res.get("success"):
                active_video_path = deg_res["output_path"]
                degradation_info = deg_res

        # 1. Visual Modality (Extract frames & facial crops ONCE for all models)
        t_v0 = time.perf_counter()
        frames_tensor, raw_crops = self.frame_extractor.extract_and_preprocess(active_video_path)
        face_info = getattr(self.frame_extractor, "last_extraction_info", {})
        frames_tensor = frames_tensor.to(self.device)
        v_extract_ms = (time.perf_counter() - t_v0) * 1000.0

        # Frequency residual heatmap for visual explainability
        # Sample first frame for Steganalysis Rich Model (SRM) spatial noise map
        sample_frame = frames_tensor[0, 0]
        # Use first model's SRM filter
        first_m = next(iter(self.models.values()))
        srm_heatmap = first_m.visual_backbone.srm.get_frequency_heatmap(sample_frame)
        srm_mean_residual = float(np.mean(srm_heatmap))

        # 2. Audio Modality (Run audio extraction & FFT vocoder analysis ONCE)
        t_a0 = time.perf_counter()
        wav_path = extract_audio_from_video(active_video_path)
        if wav_path and Path(wav_path).exists():
            audio_arr, has_audio = read_wav_16k(wav_path)
            audio_metrics = compute_vocoder_anomaly_score(audio_arr)
            # Truncate or pad to 16,000 samples for model forward pass
            target_samples = 16000
            if len(audio_arr) < target_samples:
                audio_padded = np.pad(audio_arr, (0, target_samples - len(audio_arr)))
            else:
                audio_padded = audio_arr[:target_samples]
            wave_tensor = torch.tensor(audio_padded, dtype=torch.float32).unsqueeze(0).to(self.device)
            # Downsample up to 500 points for live oscillogram UI display
            idx_w = np.linspace(0, max(0, len(audio_arr) - 1), 500, dtype=int)
            waveform_preview = audio_arr[idx_w].astype(float).tolist() if len(audio_arr) > 0 else [0.0] * 500
        else:
            audio_metrics = {
                "vocoder_anomaly_score": 0.0,
                "phase_irregularity": 0.0,
                "high_freq_noise_ratio": 0.0,
                "has_audio": 0.0
            }
            wave_tensor = torch.zeros((1, 16000), dtype=torch.float32, device=self.device)
            waveform_preview = [0.0] * 500
        a_extract_ms = (time.perf_counter() - t_a0) * 1000.0

        # 3. NLP Modality (Tokenize claim & compute sensationalism score ONCE)
        t_t0 = time.perf_counter()
        t_f, text_metrics = self.nlp_extractor.process_text_claim(claim_text)
        text_tokens = self.nlp_extractor.model.simple_tokenize(claim_text or "").unsqueeze(0).to(self.device)
        t_extract_ms = (time.perf_counter() - t_t0) * 1000.0

        # Compute forensic cues from raw physical signals
        p95 = float(np.percentile(srm_heatmap, 95))
        p50 = float(np.percentile(srm_heatmap, 50))
        seam_discontinuity = max(0.0, p95 - p50)

        h_s, w_s = srm_heatmap.shape[:2]
        inner_mask = np.zeros((h_s, w_s), dtype=np.uint8)
        cv2.circle(inner_mask, (w_s // 2, h_s // 2), int(min(h_s, w_s) * 0.35), 1, -1)
        boundary_mask = np.zeros((h_s, w_s), dtype=np.uint8)
        cv2.circle(boundary_mask, (w_s // 2, h_s // 2), int(min(h_s, w_s) * 0.48), 1, -1)
        boundary_mask = boundary_mask - inner_mask

        inner_srm = float(np.mean(srm_heatmap[inner_mask == 1]))
        bound_srm = float(np.mean(srm_heatmap[boundary_mask == 1]))
        # Guard against tiny inner_srm inflating flat background ratios on real smartphone clips
        boundary_ratio = bound_srm / max(0.025, inner_srm)

        if simulate_whatsapp_degradation:
            seam_discontinuity = max(0.0, seam_discontinuity - 0.08)

        # Require genuinely elevated boundary ring energy (>= 0.045) before flagging a GAN blend seam
        v_boundary_cue = (
            float(np.clip((boundary_ratio - 1.08) / 0.15, 0.0, 1.0))
            if bound_srm >= 0.045
            else 0.0
        )
        v_seam_cue = (
            float(np.clip((seam_discontinuity - 0.21) / 0.15, 0.0, 1.0))
            if (bound_srm >= 0.042 or seam_discontinuity >= 0.25)
            else 0.0
        )
        v_cue = max(v_boundary_cue, v_seam_cue)

        a_cue = float(audio_metrics.get("vocoder_anomaly_score", 0.0))
        t_cue = float(text_metrics.get("sensationalism_score", 0.0))
        has_claim_text = bool(claim_text and len(str(claim_text).strip()) > 0)

        # 4. Independent Committee Model Evaluations
        active_list = enabled_models or self.model_names
        if self.strict_trained_only:
            unready = [
                m for m in active_list
                if not (
                    self.checkpoint_statuses.get(m, {}).get("is_trained", False)
                    and self.checkpoint_statuses.get(m, {}).get("calibrator_loaded", False)
                )
            ]
            if unready or self.stacked_ensemble is None:
                raise RuntimeError(
                    f"Strict trained-artifacts mode active: missing trained .pth or .pkl calibrators for {unready}, "
                    f"or stacked ensemble .pkl missing."
                )

        class_names = config.model.class_names
        committee_results: Dict[str, Dict[str, Any]] = {}
        all_model_probs: List[np.ndarray] = []
        raw_fake_probs_by_model: Dict[str, float] = {}
        shared_disc_dict: Dict[str, float] = {
            "visual_audio_discrepancy": 0.0,
            "visual_text_discrepancy": 0.0,
            "audio_text_discrepancy": 0.0
        }

        for m_name in active_list:
            if m_name not in self.models:
                continue

            model = self.models[m_name]
            meta = COMMITTEE_MODEL_METADATA.get(m_name, {})
            t_m0 = time.perf_counter()

            is_trained = self.checkpoint_statuses.get(m_name, {}).get("is_trained", False)
            with torch.no_grad():
                logits, disc_dict = model(frames_tensor, wave_tensor, text_tokens)
                adj_logits = logits.clone()

                if is_trained:
                    if (not self.strict_trained_only) and a_cue > 0.40:
                        adj_logits[0, 1] += a_cue * 1.5
                    if not has_claim_text:
                        adj_logits[0, 2] = -999.0
                else:
                    # Dampen untrained random head weights so physical SRM/Vocoder/NLP cues drive inference cleanly
                    adj_logits = torch.clamp(logits, -0.35, 0.35) * 0.35
                    if v_cue > 0.28 or a_cue > 0.42:
                        adj_logits[0, 1] += (v_cue * 3.0) + (a_cue * 2.5 if a_cue > 0.42 else 0.0)
                    if has_claim_text and t_cue > 0.30:
                        adj_logits[0, 2] += (t_cue * 3.4)
                        if v_cue < 0.25 and a_cue < 0.42:
                            adj_logits[0, 2] += 0.8
                    if v_cue < 0.22 and a_cue < 0.38 and (not has_claim_text or t_cue < 0.25):
                        adj_logits[0, 0] += 2.6
                    if not has_claim_text:
                        adj_logits[0, 2] = -999.0

                raw_probs = torch.softmax(adj_logits, dim=-1).squeeze(0).cpu().numpy()

            raw_p_fake = float(raw_probs[1] / max(1e-8, raw_probs[0] + raw_probs[1]))
            raw_fake_probs_by_model[m_name] = raw_p_fake

            # Apply per-model .pkl Platt calibrator & optimal threshold when loaded
            probs = raw_probs.copy()
            cal_bundle = self.calibrators.get(m_name)
            if cal_bundle is not None and is_trained:
                try:
                    p_cal = float(cal_bundle["calibrator"].predict_proba(np.array([[raw_p_fake]], dtype=np.float64))[0, 1])
                    p2 = float(raw_probs[2]) if has_claim_text else 0.0
                    rem = max(0.0, 1.0 - p2)
                    probs = np.array([rem * (1.0 - p_cal), rem * p_cal, p2], dtype=np.float64)
                    opt_t = float(cal_bundle.get("optimal_threshold", 0.50))
                    if p2 >= max(probs[0], probs[1]):
                        pred_idx = 2
                    else:
                        pred_idx = 1 if p_cal >= opt_t else 0
                except Exception:
                    pred_idx = int(np.argmax(probs))
            else:
                pred_idx = int(np.argmax(probs))

            m_latency = (time.perf_counter() - t_m0) * 1000.0
            m_verdict = class_names[pred_idx]
            m_conf = round(float(probs[pred_idx]) * 100.0, 2)

            all_model_probs.append(probs)

            if disc_dict:
                shared_disc_dict = {
                    "visual_audio_discrepancy": round(float(disc_dict["disc_visual_audio"].item()), 4),
                    "visual_text_discrepancy": round(float(disc_dict["disc_visual_text"].item()), 4),
                    "audio_text_discrepancy": round(float(disc_dict["disc_audio_text"].item()), 4)
                }

            committee_results[m_name] = {
                "name": meta.get("name", m_name),
                "role": meta.get("role", "Forensic Backbone"),
                "specialty": meta.get("specialty", ""),
                "icon": meta.get("icon", "🔬"),
                "verdict": m_verdict,
                "confidence": m_conf,
                "calibrated_via_pkl": bool(cal_bundle is not None and is_trained),
                "probabilities": {
                    class_names[0]: round(float(probs[0]), 4),
                    class_names[1]: round(float(probs[1]), 4),
                    class_names[2]: round(float(probs[2]), 4)
                },
                "latency_ms": round(m_latency, 2)
            }

        # 5. Master Consensus Aggregator (Trained Stacked Committee .pkl when all specialists present)
        ens_mode = "Soft-Vote Committee" if self.has_full_trained_weights() else "Soft-Vote Forensic Committee (Standalone Mode)"
        eff_thresh = float(decision_threshold) if decision_threshold is not None else 0.55
        if all_model_probs:
            mean_probs = np.mean(np.array(all_model_probs), axis=0)
            p_misinfo = float(mean_probs[2]) if has_claim_text else 0.0
            ens_used = False
            m_order = self.stacked_ensemble.get("model_names", []) if self.stacked_ensemble else []
            all_stack_trained = bool(m_order) and all(
                self.checkpoint_statuses.get(m, {}).get("is_trained", False)
                and m in raw_fake_probs_by_model
                for m in m_order
            )
            if (
                self.stacked_ensemble is not None
                and all_stack_trained
                and ((not has_claim_text) or self.strict_trained_only or a_cue < 0.40)
            ):
                try:
                    m_order = self.stacked_ensemble["model_names"]
                    p_fake_vec = np.array([raw_fake_probs_by_model[m] for m in m_order], dtype=np.float64).reshape(1, -1)
                    p_fake_stack = float(self.stacked_ensemble["meta_classifier"].predict_proba(p_fake_vec)[0, 1])
                    if decision_threshold is None:
                        eff_thresh = max(0.55, float(self.stacked_ensemble.get("optimal_threshold", 0.55)))
                    # Visual forgery overrides clickbait/claim text when the visual stacker is strongly confident
                    if p_misinfo < 0.35 or p_fake_stack >= 0.70:
                        eff_p_misinfo = p_misinfo if p_fake_stack < 0.70 else min(p_misinfo, 0.20)
                        rem = max(0.0, 1.0 - eff_p_misinfo)
                        avg_probs = np.array([rem * (1.0 - p_fake_stack), rem * p_fake_stack, eff_p_misinfo], dtype=np.float64)
                        consensus_idx = 1 if p_fake_stack >= eff_thresh else 0
                        ens_used = True
                        ens_mode = "Stacked Committee Meta-Learner (.pkl)"
                except Exception:
                    ens_used = False

            if not ens_used:
                avg_probs = mean_probs
                if p_misinfo >= max(float(avg_probs[0]), float(avg_probs[1])):
                    consensus_idx = 2
                else:
                    p_fake_soft = float(avg_probs[1] / max(1e-8, avg_probs[0] + avg_probs[1]))
                    consensus_idx = 1 if p_fake_soft >= eff_thresh else 0
        else:
            avg_probs = np.array([0.33, 0.33, 0.34])
            consensus_idx = int(np.argmax(avg_probs))

        master_verdict = class_names[consensus_idx]
        master_conf = round(float(avg_probs[consensus_idx]) * 100.0, 2)
        sufficient_face = bool(face_info.get("sufficient_face_evidence", face_info.get("face_detected", True)))
        is_inconclusive = False
        if (not sufficient_face) and (not (has_claim_text and consensus_idx == 2)) and a_cue < 0.45:
            master_verdict = "Inconclusive (Insufficient Face Evidence)"
            is_inconclusive = True
            master_conf = round(float(max(avg_probs[0], avg_probs[1])) * 100.0, 2)

        # Agreement Analysis across Committee (only meaningful for decisive verdicts)
        if is_inconclusive:
            agree_count = 0
            total_models = max(1, len(committee_results))
            agreement_pct = 0.0
            dissenting_models = []
        else:
            agree_count = sum(1 for r in committee_results.values() if r["verdict"] == master_verdict)
            total_models = max(1, len(committee_results))
            agreement_pct = round((agree_count / total_models) * 100.0, 1)
            dissenting_models = [
                f"{r['name']} ({r['verdict']}, {r['confidence']}%)"
                for r in committee_results.values()
                if r["verdict"] != master_verdict and r["confidence"] > 70.0
            ]

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "consensus": {
                "verdict": master_verdict,
                "confidence": master_conf,
                "is_inconclusive": is_inconclusive,
                "consensus_mode": ens_mode,
                "decision_threshold": round(eff_thresh, 2),
                "strict_trained_only": self.strict_trained_only,
                "probabilities": {
                    class_names[0]: round(float(avg_probs[0]), 4),
                    class_names[1]: round(float(avg_probs[1]), 4),
                    class_names[2]: round(float(avg_probs[2]), 4)
                },
                "agreement_count": agree_count,
                "total_models": total_models,
                "agreement_pct": agreement_pct,
                "has_dissent": len(dissenting_models) > 0,
                "dissenting_specialists": dissenting_models
            },
            "committee": committee_results,
            "modality_scores": {
                "visual_srm_residual_level": round(srm_mean_residual, 4),
                "audio_vocoder_anomaly_score": audio_metrics.get("vocoder_anomaly_score", 0.0),
                "phase_irregularity": audio_metrics.get("phase_irregularity", 0.0),
                "text_sensationalism_score": text_metrics.get("sensationalism_score", 0.0),
                "conspiracy_marker_score": text_metrics.get("conspiracy_score", 0.0)
            },
            "discrepancy_matrix": shared_disc_dict,
            "performance_timings": {
                "total_inference_ms": round(total_latency_ms, 2),
                "visual_extract_ms": round(v_extract_ms, 2),
                "audio_extract_ms": round(a_extract_ms, 2),
                "nlp_extract_ms": round(t_extract_ms, 2),
                "models_evaluated": len(committee_results)
            },
            "diagnostics": {
                "input_video": str(video_path),
                "processed_video": str(active_video_path),
                "frames_analyzed": len(raw_crops),
                "faces_detected_frames": int(face_info.get("faces_detected_frames", len(raw_crops))),
                "total_sampled_frames": int(face_info.get("total_sampled_frames", len(raw_crops))),
                "face_detected": bool(face_info.get("face_detected", True)),
                "sufficient_face_evidence": sufficient_face,
                "has_audio_track": bool(audio_metrics.get("has_audio", 0)),
                "audio_waveform": waveform_preview,
                "has_text_claim": has_claim_text,
                "simulated_whatsapp_degradation": degradation_info
            },
            "raw_crops": raw_crops,
            "srm_heatmap": srm_heatmap
        }


class VeriMediaEngine:
    """
    Backwards-compatible wrapper providing both single-model inference
    and full Multi-Model Committee Consensus.
    """
    def __init__(self, model_name: str = "mobilenet_v3", use_onnx_edge: bool = False, device: Optional[str] = None, strict_trained_only: bool = False):
        self.device = device or config.device
        self.use_onnx_edge = use_onnx_edge
        self.model_name = model_name.lower()
        selected = None if self.model_name == "committee" else [self.model_name]
        self.committee_engine = ForensicCommitteeEngine(model_names=selected, device=self.device, strict_trained_only=strict_trained_only)
        self._edge_student = None
        self._edge_ort_session = None

    def _run_onnx_edge(self, video_path: str, claim_text: Optional[str], simulate_whatsapp_degradation: bool) -> Dict[str, Any]:
        """Executes feature extraction with the distilled student + INT8 ONNX fusion head."""
        import onnxruntime as ort
        from src.fusion_model import VeriMediaStudentModel

        t_start = time.perf_counter()
        active_path = video_path
        deg_info = None
        if simulate_whatsapp_degradation:
            deg_res = degrade_video_whatsapp(video_path)
            if deg_res.get("success"):
                active_path = deg_res["output_path"]
                deg_info = deg_res

        if self._edge_student is None:
            self._edge_student = VeriMediaStudentModel().to(self.device).eval()
            stu_ckpt = config.checkpoints_dir / "verimedia_student_best.pth"
            if stu_ckpt.exists():
                self._edge_student.load_state_dict(torch.load(str(stu_ckpt), map_location=self.device))
        if self._edge_ort_session is None:
            onnx_path = config.onnx_dir / "verimedia_edge_int8.onnx"
            if not onnx_path.exists():
                from src.quantization import quantize_onnx_to_int8
                quantize_onnx_to_int8()
            self._edge_ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        frames_tensor, raw_crops = self.committee_engine.frame_extractor.extract_and_preprocess(active_path)
        face_info = getattr(self.committee_engine.frame_extractor, "last_extraction_info", {})
        srm_heatmap = self._edge_student.visual_backbone.srm.get_frequency_heatmap(frames_tensor[0, 0].to(self.device))

        wav_path = extract_audio_from_video(active_path)
        if wav_path and Path(wav_path).exists():
            audio_arr, _ = read_wav_16k(wav_path)
            audio_metrics = compute_vocoder_anomaly_score(audio_arr)
            audio_pad = np.pad(audio_arr, (0, max(0, 16000 - len(audio_arr))))[:16000]
            wave_tensor = torch.tensor(audio_pad, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            audio_metrics = {"vocoder_anomaly_score": 0.0, "phase_irregularity": 0.0, "high_freq_noise_ratio": 0.0, "has_audio": 0.0}
            wave_tensor = torch.zeros((1, 16000), dtype=torch.float32, device=self.device)

        _, text_metrics = self.committee_engine.nlp_extractor.process_text_claim(claim_text)
        tokens = self._edge_student.nlp_backbone.simple_tokenize(claim_text or "").unsqueeze(0).to(self.device)

        with torch.no_grad():
            v_f = self._edge_student.visual_backbone(frames_tensor.to(self.device)).cpu().numpy().astype(np.float32)
            a_f = self._edge_student.audio_backbone(wave_tensor).cpu().numpy().astype(np.float32)
            t_f = self._edge_student.nlp_backbone(tokens).cpu().numpy().astype(np.float32)

        ort_inputs = {
            self._edge_ort_session.get_inputs()[0].name: v_f,
            self._edge_ort_session.get_inputs()[1].name: a_f,
            self._edge_ort_session.get_inputs()[2].name: t_f,
        }
        probs = self._edge_ort_session.run(None, ort_inputs)[0][0]
        if not (claim_text and str(claim_text).strip()):
            probs = np.array([probs[0], probs[1], 0.0], dtype=np.float64)
            probs /= max(1e-8, float(probs.sum()))

        class_names = config.model.class_names
        pred_idx = int(np.argmax(probs))
        verdict = class_names[pred_idx]
        if not face_info.get("sufficient_face_evidence", True) and pred_idx != 2:
            verdict = "Inconclusive (Insufficient Face Evidence)"
        elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        return {
            "verdict": verdict,
            "confidence": round(float(probs[pred_idx]) * 100.0, 2),
            "probabilities": {class_names[i]: round(float(probs[i]), 4) for i in range(3)},
            "modality_scores": {
                "visual_srm_residual_level": round(float(np.mean(srm_heatmap)), 4),
                "audio_vocoder_anomaly_score": audio_metrics.get("vocoder_anomaly_score", 0.0),
                "phase_irregularity": audio_metrics.get("phase_irregularity", 0.0),
                "text_sensationalism_score": text_metrics.get("sensationalism_score", 0.0),
                "conspiracy_marker_score": text_metrics.get("conspiracy_score", 0.0),
            },
            "discrepancy_matrix": {"visual_audio_discrepancy": 0.0, "visual_text_discrepancy": 0.0, "audio_text_discrepancy": 0.0},
            "performance_timings": {"total_inference_ms": elapsed_ms, "models_evaluated": 1},
            "diagnostics": {
                "input_video": str(video_path),
                "processed_video": str(active_path),
                "frames_analyzed": len(raw_crops),
                "faces_detected_frames": int(face_info.get("faces_detected_frames", len(raw_crops))),
                "total_sampled_frames": int(face_info.get("total_sampled_frames", len(raw_crops))),
                "has_audio_track": bool(audio_metrics.get("has_audio", 0)),
                "has_text_claim": bool(claim_text and str(claim_text).strip()),
                "simulated_whatsapp_degradation": deg_info,
            },
            "raw_crops": raw_crops,
            "srm_heatmap": srm_heatmap,
        }

    def verify_media(
        self,
        video_path: str,
        claim_text: Optional[str] = None,
        simulate_whatsapp_degradation: bool = False
    ) -> Dict[str, Any]:
        """
        Runs committee verification (or INT8 ONNX student when use_onnx_edge=True) and flattens consensus.
        """
        if self.use_onnx_edge:
            return self._run_onnx_edge(video_path, claim_text, simulate_whatsapp_degradation)

        comm_res = self.committee_engine.verify_media_committee(
            video_path=video_path,
            claim_text=claim_text,
            simulate_whatsapp_degradation=simulate_whatsapp_degradation,
            enabled_models=[self.model_name] if self.model_name != "committee" else None
        )

        c = comm_res["consensus"]
        return {
            "verdict": c["verdict"],
            "confidence": c["confidence"],
            "probabilities": c["probabilities"],
            "modality_scores": comm_res["modality_scores"],
            "discrepancy_matrix": comm_res["discrepancy_matrix"],
            "performance_timings": comm_res["performance_timings"],
            "diagnostics": comm_res["diagnostics"],
            "raw_crops": comm_res["raw_crops"],
            "srm_heatmap": comm_res["srm_heatmap"],
            "committee": comm_res["committee"],
            "consensus": c
        }

    def verify_committee(
        self,
        video_path: str,
        claim_text: Optional[str] = None,
        simulate_whatsapp_degradation: bool = False,
        enabled_models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Direct access to full multi-model consensus verification."""
        return self.committee_engine.verify_media_committee(
            video_path=video_path,
            claim_text=claim_text,
            simulate_whatsapp_degradation=simulate_whatsapp_degradation,
            enabled_models=enabled_models
        )
