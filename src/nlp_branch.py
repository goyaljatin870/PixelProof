"""
nlp_branch.py - Contextual & Regional Claim Verification Backbone
Tokenizes regional dialect transcripts and video captions, analyzes claims
for contextual deception, and outputs a 768-dimensional textual intent vector (T_f).
"""

import re
import zlib
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional, List


# Deceptive claim indicators common in regional election/dark social campaigns
MISINFORMATION_LEXICON = {
    "urgency": ["breaking", "urgent", "must watch", "forward this", "before deleted", "share now", "viral"],
    "conspiracy": ["secret", "exposed", "hidden truth", "they don't want you to know", "leaked", "conspiracy", "scandal"],
    "fabricated_electoral": ["rigged", "vote stolen", "bribe", "resignation leaked", "caught on tape", "election hacked", "scam"]
}


def analyze_claim_sensationalism(text: str) -> Dict[str, float]:
    """
    Computes heuristic forensic claim scores based on regional disinformation markers.
    """
    if not text or not text.strip():
        return {
            "sensationalism_score": 0.0,
            "conspiracy_score": 0.0,
            "urgency_score": 0.0,
            "has_claim_text": 0.0
        }

    clean_text = text.lower()
    total_words = max(1, len(clean_text.split()))

    urgency_hits = sum(1 for phrase in MISINFORMATION_LEXICON["urgency"] if phrase in clean_text)
    conspiracy_hits = sum(1 for phrase in MISINFORMATION_LEXICON["conspiracy"] if phrase in clean_text)
    electoral_hits = sum(1 for phrase in MISINFORMATION_LEXICON["fabricated_electoral"] if phrase in clean_text)

    # Exclamation mark density
    exclamation_count = clean_text.count("!")
    exclamation_density = min(1.0, exclamation_count / 5.0)

    urgency_score = min(1.0, (urgency_hits * 0.4) + (exclamation_density * 0.3))
    conspiracy_score = min(1.0, (conspiracy_hits * 0.5) + (electoral_hits * 0.6))
    
    overall_score = float(np.clip(0.5 * conspiracy_score + 0.3 * urgency_score + 0.2 * exclamation_density, 0.0, 1.0))

    return {
        "sensationalism_score": round(overall_score, 4),
        "conspiracy_score": round(conspiracy_score, 4),
        "urgency_score": round(urgency_score, 4),
        "has_claim_text": 1.0
    }


class IntentClaimBackbone(nn.Module):
    """
    NLP Intent & Contextual Claim Neural Backbone.
    Tokenizes text input and processes it through self-attention layers to produce
    a 768-dimensional textual-intent representation vector (T_f).
    Compliant with on-device edge execution without requiring gigabytes of model weights.
    """
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 256, out_dim: int = 768, num_heads: int = 4):
        super().__init__()
        self.out_dim = out_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Positional encoding projection
        self.pos_encoder = nn.Parameter(torch.randn(1, 128, embed_dim) * 0.02)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Projection from embedding dim to 768 (matching XLM-RoBERTa CLS dimension)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

    def simple_tokenize(self, text: str, max_len: int = 128) -> torch.Tensor:
        """
        Deterministic lightweight hash tokenization for edge deployment.
        """
        words = re.findall(r"\w+", text.lower()) if text else []
        tokens = [1]  # [CLS] token
        for w in words:
            # Deterministic CRC32 hash to vocab space [2, 9999] across all platforms/processes
            tok = (zlib.crc32(w.encode("utf-8")) % 9998) + 2
            tokens.append(tok)
            if len(tokens) >= max_len - 1:
                break
        tokens.append(2)  # [SEP] token
        
        # Pad to max_len
        pad_len = max_len - len(tokens)
        if pad_len > 0:
            tokens.extend([0] * pad_len)
            
        return torch.tensor(tokens, dtype=torch.long)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            input_ids: (B, seq_len)
        Returns:
            T_f: (B, 768) sequence embedding (simulating [CLS] token)
        """
        seq_len = input_ids.shape[1]
        x = self.embedding(input_ids)  # (B, seq_len, embed_dim)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        hidden = self.transformer(x)  # (B, seq_len, embed_dim)
        cls_hidden = hidden[:, 0, :]  # [CLS] representation (B, embed_dim)
        t_f = self.proj(cls_hidden)   # (B, 768)
        return t_f


class NLPBranchExtractor:
    """
    High-level extractor for video narrative context, captions, and claims.
    """
    def __init__(self, model: Optional[IntentClaimBackbone] = None):
        self.model = model if model is not None else IntentClaimBackbone(out_dim=768)
        self.model.eval()

    def process_text_claim(self, claim_text: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Processes claim text or caption, computes sensationalism metrics, and returns:
            t_f_tensor: (1, 768)
            claim_metrics: dict of forensic text scores
        """
        text = claim_text if claim_text is not None else ""
        metrics = analyze_claim_sensationalism(text)
        
        input_ids = self.model.simple_tokenize(text).unsqueeze(0)  # (1, 128)
        with torch.no_grad():
            t_f = self.model(input_ids)
            
        return t_f, metrics
