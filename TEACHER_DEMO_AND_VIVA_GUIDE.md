# VeriMedia — Teacher Demo & Viva Presentation Guide

## 1. 30-Second Project Pitch (What to Say First)
> *"VeriMedia is an edge-optimized multimodal deepfake and regional misinformation detection engine designed for heavily compressed dark-social media like WhatsApp and Telegram. Instead of relying only on visual pixels—which break under H.264 CRF 38 compression and miss real videos paired with fake captions—VeriMedia fuses **3 forensic streams**: Steganalysis Rich Model (SRM) high-pass visual noise filters, 16 kHz acoustic vocoder inspection, and regional claim intent modeling. Our 6-model RTX 4050 GPU-trained committee achieves **0.9674 ROC-AUC (90.25% balanced accuracy, 95.3% real recall)** on the 1,092-video identity-disjoint FaceForensics++ C23 test split, and we distill the ResNet-50 teacher down into a **0.92 MB INT8 ONNX mobile student** running in sub-millisecond CPU inference."*

---

## 2. Live Demo Checklist (5 Minutes in Front of Your Teacher)

### Step A: Launch the Web Dashboard (1 Click)
- Double-click **`Start_Dashboard.bat`** (or run `.venv\Scripts\python.exe -m streamlit run frontend/app.py`).
- It opens automatically at `http://localhost:8501`.

### Step B: Walk Through the 5 Dashboard Tabs
1. **Tab 1 — Forensic Committee Scanner** (Main Demo):
   - Click **`🏛️ Authentic Briefing`** → shows clean facial crops, normal SRM residual distribution, clean audio stream, and **`AUTHENTIC`** consensus.
   - Click **`🎭 Face Swap Deepfake`** → shows **`MANIPULATED PAYLOAD`**, inspect the **SRM Noise Heatmap** sub-tab (highlights manipulation boundary seams) and **Anomaly Sensitivity Polar Radar**.
   - Click **`📱 WhatsApp Viral Hoax`** → turns on live **H.264 CRF 38 + 15 FPS WhatsApp compression** via FFmpeg + sensational election claim → inspects the **Inter-Modality Discrepancy Matrix** ($V_f \times A_f$, $V_f \times T_f$, $A_f \times T_f$).
   - Click **`🎙️ Voice Cloned Video`** → demonstrates acoustic **Vocoder Anomaly Score** & phase irregularity (>6 kHz high-frequency vocoder artifacts) alongside urgent claim contradiction.
   - Open **`📜 Forensic Verification Certificate`** sub-tab → show the SHA-256 verifiable JSON dossier download.
2. **Tab 2 — Evaluation Metrics & Confusion Matrices**:
   - Point out the **1,092 unseen test videos** (identity-disjoint actors).
   - Compare **Stacked Committee (.pkl)** (`0.9674 AUC`, `89.02%` raw acc / `90.25%` bal acc, and `95.3%` real recall with only `7/149` false positives at calibrated threshold `0.61`) vs **Distilled Student** (`0.8935 AUC`, `82.84% Acc`) vs individual backbones (`ResNet-50` `0.9516`, `EfficientNetV2` `0.9408`, `XceptionNet` `0.9203`, `MobileNetV3` `0.9046`, `Temporal-LSTM` `0.8958`, `MesoInception-4` `0.6410`).
   - Show the **6-Epoch Training Convergence Curves** and **Per-Category Accuracy** across all 7 FaceForensics++ sequences (`original`, `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`, `FaceShifter`, `DeepFakeDetection`).
3. **Tab 3 — Trained Artifacts (.pth & .pkl)**:
   - Show the `.pkl` Platt calibrators and the learned **Stacked Meta-Learner weights** (e.g., `ResNet50`, `MesoNet`, `EfficientNetV2`, `TemporalLSTM`, `MobileNetV3`, `Xception`) plus hot-reload support.
4. **Tab 4 — Edge Verification Matrix**:
   - Click **`🚀 Run Live 25-Iteration Benchmark`** → proves real-time compliance:
     - Model Size: **0.92 MB** (target `<= 45 MB`)
     - Mean CPU Latency: **< 1 ms** (target `< 300 ms`)
     - Peak Active RAM Delta: **< 5 MB** (target `< 180 MB`)
5. **Tab 5 — Dataset Manifest Analytics**:
   - Shows the **7,000 FaceForensics++ C23 videos** (`1,000 REAL` / `6,000 FAKE`), `1,028 unique actor identities`, and identity-disjoint split (`4,692 train / 1,216 val / 1,092 test`).

### Step C: Optional Terminal / Code Verification (1 Click)
- Double-click **`Run_CLI_Demo.bat`** to show the end-to-end CLI pipeline (`quantize`, `benchmark`, `predict`, and all **13/13 PyTest unit tests passing**).

---

## 3. Architecture & Key Numbers Cheat-Sheet

| Component | Key Implementation (`src/`) | Key Fact / Dimension |
| :--- | :--- | :--- |
| **Visual Front-End** | `src/srm_filter.py`, `src/visual_branch.py` | 3 fixed `5x5` zero-sum SRM high-pass kernels (clamped `[-3, +3]`), blended `x + 0.3 * residuals`; Haar face crops `224x224` |
| **6-Model Zoo** | `src/models_zoo.py` | `MobileNetV3` (576d), `Temporal-LSTM` (512d), `MesoInception-4` (512d), `ResNet-50` (1024d), `XceptionNet` (1024d), `EfficientNetV2-S` (1280d) |
| **Audio Branch** | `src/audio_branch.py` | 16 kHz mono PCM -> 4-layer 1D Conv + 2-layer BiGRU -> `768d` + FFT high-freq (>6 kHz) & phase jitter score |
| **NLP Branch** | `src/nlp_branch.py` | Deterministic CRC32 hash tokenizer (`vocab=10,000`, `len=128`) + 2-layer Transformer -> `768d` + lexicon scores |
| **Late-Fusion Head** | `src/fusion_model.py` | LayerNorm per modality + cosine discrepancy (`V-A`, `V-T`, `A-T`) + 3-layer residual MLP -> 3 classes |
| **Ensemble & KD** | `src/train_zoo.py`, `src/distillation.py` | Platt calibration (`.pkl`) + Logistic Regression stacker (`0.9674 AUC`); ResNet-50 -> MobileNetV3 KD (`alpha=0.5`, `T=2.0`, `0.8935 AUC`) |
| **Edge Quantization** | `src/quantization.py`, `src/benchmark.py` | ONNX export (`3.53 MB` FP32) -> QUInt8 dynamic PTQ (`0.92 MB` INT8, **3.83x compression**) |

---

## 4. Likely Teacher Viva Questions & Strong Answers

1. **Why do you call it "Multimodal" instead of just using a face CNN?**
   > *A visual CNN alone cannot detect **"cheapfakes" (Class 2: Contextual Misinformation)**—real authentic videos shared with fabricated viral captions. Adding the audio vocoder branch and NLP claim branch lets our Late-Fusion Discrepancy Matrix detect cross-modal contradictions ($V_f \times T_f$, $V_f \times A_f$).*

2. **What is the SRM filter and why use fixed weights?**
   > *SRM stands for **Steganalysis Rich Model** (`src/srm_filter.py`). Its three `5x5` zero-sum high-pass kernels suppress visual color/identity semantics and isolate microscopic boundary blending seams and compression noise residuals. Keeping the kernels fixed (`requires_grad=False`) prevents the network from memorizing faces and forces it to inspect high-frequency forensic artifacts.*

3. **How did you prevent data leakage on FaceForensics++?**
   > *FaceForensics++ uses 1,000 actor IDs across both original (`XXX.mp4`) and manipulated (`XXX_YYY.mp4`) clips. If split randomly by clip, the same actor appears in both train and test. Our `src/inspect_dataset.py` groups strictly by `source_id` so all 1,028 identities are **identity-disjoint** across `train` (4,692), `val` (1,216), and `test` (1,092).*

4. **Why train 6 different models instead of 1?**
   > *Each architecture captures complementary cues: `EfficientNetV2-S` and `ResNet-50` capture multi-scale spatial texture, `XceptionNet` specializes in separable spatial blending seams, `Temporal-LSTM` catches frame-to-frame blinking/flickering jitter, `MesoInception-4` targets mesoscopic compression blocks, and `MobileNetV3` provides mobile speed. Combining them via the balanced **Stacked Meta-Learner (`.pkl`)** boosts test ROC-AUC to **0.9674** and reduces authentic false positives to **7/149 (95.3% real recall)**.*

5. **How do you deploy a deep model onto a low-end phone (< 45 MB, < 300 ms)?**
   > *Through **Cross-Modal Knowledge Distillation** (`ResNet-50` teacher -> `MobileNetV3` student using KL-divergence `T=2.0` + discrepancy MSE alignment, achieving `0.8935` ROC-AUC under mixed clean+compressed evaluation) followed by **ONNX 8-bit Dynamic Quantization (QUInt8)**, shrinking the fusion payload from `3.54 MB` to **`0.92 MB`** with sub-millisecond CPU inference latency.*

6. **What happens during WhatsApp forwarding simulation?**
   > *`src/degradation_pipeline.py` uses FFmpeg to transcode inbound video to `480p` max resolution, `15 FPS`, H.264 `CRF 38` (heavy quantization), and `16 kbps` mono audio—reproducing real messaging platform compression before forensic inspection.*

7. **What are the honest limitations of the current system?**
   > *(1) **FaceForensics++ C23 has no spoken audio tracks** (`had_audio=False`), so real-world voice cloning co-training needs an audio-visual dataset like FakeAVCeleb; (2) **NeuralTextures** mouth-only edits are hardest (`66.44%` vs `91.95%+` on Deepfakes/Face2Face and `100%` on DeepFakeDetection); (3) Real-world out-of-distribution phone videos (e.g. vertical 60fps iPhone `.MOV` HDR sharpening) differ from YouTube FF++ framing and can benefit from training on in-the-wild / smartphone authentic corpora alongside FF++.*

8. **How does the system verify trained weights vs fallback mode?**
   > *With all 7 `.pth` checkpoints and 7 `.pkl` calibrator/ensemble bundles present under `artifacts/checkpoints/`, the dashboard enforces **`strict_trained_only=True`** (`Trained Committee Online`). If `.pth` files are removed (e.g., on a clean GitHub clone), it gracefully falls back to Standalone Forensic Mode (`SRM + Vocoder FFT + NLP + .pkl`).*
