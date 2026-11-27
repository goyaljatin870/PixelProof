# PixelProof: Multimodal Deepfake & Regional Misinformation Detection Engine
> **Edge-Optimized Verification Engine for Dark-Social, Heavily Compressed Media (WhatsApp / Telegram)**

---

## 1. Executive Summary & Pitch

**PixelProof** is an edge-optimized multimodal deepfake and regional misinformation detection engine engineered specifically to identify synthetic media and contextual narrative deception in heavily compressed, low-bandwidth distribution networks (WhatsApp, Telegram, regional social platforms).

Rather than relying on cloud-bound computing overhead, PixelProof operates an asynchronous late-fusion ensemble across visual, vocal, and textual modalities and applies post-training INT8 quantization to achieve sub-millisecond edge CPU inference with a total model payload of **0.92 MB** (well below the 45 MB mobile ceiling).

---

## 2. System Verification Matrix Results

Tested against the project technical specification matrix:

| Metric | Target Specification | Achieved Measured Value | Compliance Status |
| :--- | :--- | :--- | :--- |
| **Inference Latency** | `< 300 ms` per media item | **0.16 ms mean (0.24 ms p95)** | **PASS ✅** |
| **Model Size** | `≤ 45 MB` total binary | **0.92 MB** (FP32: 3.54 MB, `3.83x`) | **PASS ✅** |
| **Peak Active Memory** | `< 180 MB` active RAM | **0.46 MB** active delta | **PASS ✅** |
| **Quantization Scheme** | 8-Bit Dynamic PTQ (QUInt8) | ONNX Runtime Mobile Compatible | **PASS ✅** |

### RTX 4050 Trained Committee Results (FaceForensics++ C23 Held-Out Test Split, `N = 1,092`)
Trained across `7,000` videos (`4,692` train / `1,216` val / `1,092` identity-disjoint test across `1,028` actors) with domain-compression augmentation:

| Model / Ensemble | Test ROC-AUC | Raw Acc @0.50 | Balanced Acc | Real Recall | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Stacked Committee (`.pkl`)** | **0.9674** | **89.02%** | **90.25%** | **91.95%** (`95.3%` @0.61) | Logistic Regression meta-learner |
| **Soft-Vote Committee** | **0.9587** | **89.11%** | **88.89%** | **88.59%** (`92.62%` @0.55) | Unweighted 6-model probability mean |
| **ResNet-50** | 0.9516 | 91.14% | 87.53% | — | Deep spatial residual backbone |
| **EfficientNetV2-S** | 0.9408 | 82.20% | 86.29% | — | Compound Fused-MBConv backbone |
| **XceptionNet** | 0.9203 | 82.56% | 83.97% | — | Separable conv blending seam detector |
| **MobileNetV3-Edge** | 0.9046 | 87.64% | 77.32% | — | Lightweight spatial-frequency baseline |
| **Temporal-LSTM** | 0.8958 | 77.21% | 82.28% | — | 2D CNN + BiLSTM frame dynamics |
| **MesoInception-4** | 0.6410 | 73.43% | 56.67% | — | Shallow mesoscopic noise baseline |
| **Distilled Edge Student** | **0.8935** | **82.84%** | **81.03%** | 78.52% | `ResNet-50` → `MobileNetV3` KD (`0.92 MB` INT8) |

---

## 3. Multimodal Late-Fusion Architecture

```text
                  +-----------------------------------+
                  |      Inbound Media Payload        |
                  +-----------------+-----------------+
                                    |
            +-----------------------+-----------------------+
            |                       |                       |
            v                       v                       v
    +---------------+       +---------------+       +---------------+
    | Visual Branch |       | Audio Branch  |       |  NLP Branch   |
    | SRM Filter +  |       |  Acoustic     |       | Intent-Claim  |
    |  MobileNetV3  |       |  Vocoder Net  |       |  Transformer  |
    +---------------+       +---------------+       +---------------+
            |                       |                       |
            | Spatial-Frequency     | Acoustic Vector       | Textual Intent
            | Features (V_f: 576d)  | (A_f: 768d)           | (T_f: 768d)
            +-----------------------+-----------------------+
                                    |
                                    v
                        +-----------------------+
                        |  Modality LayerNorm   |
                        +-----------------------+
                                    |
                                    v
                        +-----------------------+
                        | Inter-Modality Cross- |
                        | Discrepancy Matrix    |
                        | (V_f x A_f, V_f x T_f)|
                        +-----------------------+
                                    |
                                    v
                        +-----------------------+
                        |   Late Fusion MLP     |
                        | 3-Layer Residual Head |
                        +-----------------------+
                                    |
                                    v
                        +-----------------------+
                        | Probability Vector    |
                        | - Authentic (P0)      |
                        | - Manipulated (P1)    |
                        | - Contextual Mis (P2) |
                        +-----------------------+
```

### Core Modalities
1. **Visual Branch (Spatial & Frequency Artifact Extraction)**:
   - Evaluates video frames sampled at 5 FPS.
   - Steganalysis Rich Model (SRM) high-pass convolutional filtering preserves subtle noise boundaries and manipulation seams even after aggressive lossy compression.
2. **Audio Branch (Acoustic Vocoder Synthesis Detection)**:
   - Extracts 16kHz mono PCM stream.
   - Computes phase irregularities, formant transitions, and high-frequency vocoder noise ratios.
3. **NLP Branch (Regional Contextual Claim Verification)**:
   - Processes regional dialect captions and speech transcripts.
   - Evaluates sensationalism, conspiracy markers, and urgency signals.
4. **Late-Fusion Discrepancy Engine**:
   - Computes explicit contradiction matrices $(V_f \times A_f^\top, V_f \times T_f^\top, A_f \times T_f^\top)$ to flag out-of-context audio dubs and deceptive captions.

---

## 4. Dataset Integration (FaceForensics++ C23 & Bundled Sample Corpus)

The project supports both the full FaceForensics++ C23 corpus (via `FFPP_ROOT_DIR` in `.env`) and the local demo corpus in `data/sample_dataset/FaceForensics++`:
```text
FaceForensics++_C23/
├── csv/
│   ├── FF++_Metadata.csv        # 7,000 video records with resolutions & labels
│   ├── original.csv
│   └── Deepfakes.csv ...
├── original/                    # 1,000 authentic pristine videos
├── Deepfakes/                   # 1,000 deepfake videos
├── Face2Face/                   # 1,000 facial reenactment videos
├── FaceShifter/                 # 1,000 face swapping videos
├── FaceSwap/                    # 1,000 face replacement videos
├── NeuralTextures/              # 1,000 neural texture facial videos
└── DeepFakeDetection/           # 1,000 actor-based deepfake videos
```

---

## 5. Quick Start & CLI Usage

### Installation
Use the local `.venv` (or create one and install dependencies):
```bash
pip install -r requirements.txt
```

### 1. Dataset Manifest Generation
Scan videos and enforce identity-disjoint splits:
```bash
python cli.py inspect
```

### 2. WhatsApp Synthetic Degradation Pipeline
Simulate WhatsApp lossy compression (480p, 15 FPS, H.264 CRF 38, 16kbps audio):
```bash
python cli.py degrade "data/sample_dataset/FaceForensics++/manipulated_sequences/Deepfakes/c23/videos/000_001.mp4"
```

### 3. Edge ONNX Export & 8-Bit Dynamic Quantization
Export the student late-fusion model to FP32 ONNX and quantize to 8-bit QUInt8:
```bash
python cli.py quantize
```

### 4. System Verification Matrix Benchmark
Evaluate latency, memory footprint, and model payload compliance:
```bash
python cli.py benchmark
```

### 5. Multimodal Media Verification
Run forensic inference on any video with optional regional claim text:
```bash
python cli.py predict "data/sample_dataset/FaceForensics++/original_sequences/youtube/c23/videos/000.mp4" --claim "Official speech from parliament" --committee
```

### 6. Interactive Web Dashboard & 1-Click Launchers
- **Web Dashboard:** Double-click `Start_Dashboard.bat` or run:
  ```bash
  streamlit run frontend/app.py
  ```
- **End-to-End CLI Demo & Tests:** Double-click `Run_CLI_Demo.bat` or run:
  ```bash
  python run_all.py --demo
  ```
- **Teacher Demo & Viva Guide:** See [TEACHER_DEMO_AND_VIVA_GUIDE.md](TEACHER_DEMO_AND_VIVA_GUIDE.md).

> **Note on Large Model Weights:** Heavy `.pth` checkpoints (~560 MB total) and raw `.mp4` videos are excluded via `.gitignore` for GitHub compatibility. When `.pth` weights are present in `artifacts/checkpoints/`, the dashboard runs in strict `.pth + .pkl` committee mode; on a fresh clone without `.pth` files, it gracefully enters Standalone Forensic Mode (SRM + Vocoder FFT + NLP + `.pkl` calibrators).

---

## 6. Automated Test Suite

Run the full pytest suite:
```bash
python -m pytest tests/
```
All 13 test suites validate SRM kernels, visual backbones, audio vocoder feature extraction, NLP claim embeddings, late fusion discrepancy calculations, and ONNX INT8 edge compliance.
