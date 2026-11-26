"""
Unit tests for Visual, Audio, NLP branches and Late-Fusion Discrepancy Head.
"""

import torch
import numpy as np
from src.visual_branch import StudentVisualBackbone
from src.audio_branch import AcousticVocoderBackbone, compute_vocoder_anomaly_score
from src.nlp_branch import IntentClaimBackbone, analyze_claim_sensationalism
from src.fusion_model import MultimodalLateFusionHead, VeriMediaStudentModel


def test_student_visual_branch():
    model = StudentVisualBackbone(out_dim=576, use_srm=True)
    model.eval()
    
    # 5 frames of shape (1, 5, 3, 224, 224)
    x = torch.rand(1, 5, 3, 224, 224)
    with torch.no_grad():
        v_f = model(x)
    assert v_f.shape == (1, 576)


def test_audio_branch():
    model = AcousticVocoderBackbone(out_dim=768)
    model.eval()
    
    # 1 second of 16kHz audio (1, 16000)
    waveform = torch.rand(1, 16000)
    with torch.no_grad():
        a_f = model(waveform)
    assert a_f.shape == (1, 768)

    # Test vocoder metric calculation
    dummy_audio = np.random.randn(16000).astype(np.float32)
    metrics = compute_vocoder_anomaly_score(dummy_audio)
    assert "vocoder_anomaly_score" in metrics
    assert "phase_irregularity" in metrics
    assert 0.0 <= metrics["vocoder_anomaly_score"] <= 1.0


def test_nlp_branch():
    model = IntentClaimBackbone(out_dim=768)
    model.eval()
    
    tokens = model.simple_tokenize("BREAKING: Leaked secret video!").unsqueeze(0)
    with torch.no_grad():
        t_f = model(tokens)
    assert t_f.shape == (1, 768)

    sens = analyze_claim_sensationalism("BREAKING: Leaked secret election footage!!")
    assert sens["sensationalism_score"] > 0.3
    assert sens["has_claim_text"] == 1.0


def test_late_fusion_and_discrepancy():
    head = MultimodalLateFusionHead(
        visual_dim=576,
        audio_dim=768,
        text_dim=768,
        num_classes=3
    )
    head.eval()

    v_f = torch.randn(2, 576)
    a_f = torch.randn(2, 768)
    t_f = torch.randn(2, 768)

    with torch.no_grad():
        logits, disc_dict = head(v_f, a_f, t_f)

    assert logits.shape == (2, 3)
    assert "disc_visual_audio" in disc_dict
    assert "disc_visual_text" in disc_dict
    assert "disc_audio_text" in disc_dict
    assert disc_dict["disc_visual_audio"].shape == (2, 1)
