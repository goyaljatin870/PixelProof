# PixelProof — Complete Teacher Viva & Cross-Examination Question Guide

> **Prepared for:** Classroom Presentation & External ML Viva Examination  
> **Repository:** `https://github.com/goyaljatin870/PixelProof`  
> **Core Stack:** PyTorch `2.14.1+cu130` (RTX 4050 `sm_89`) • FaceForensics++ C23 (`7,000` videos, `1,028` actors) • 6-Model Stacked Committee (`0.9674` Test ROC-AUC) • Distilled INT8 ONNX Edge Student (`0.92 MB`, `0.16 ms` CPU latency)

---

## 1. Quick 30-Second Viva Pitch (Memorize This First)
> *"PixelProof is an edge-optimized multimodal deepfake and regional misinformation detection engine built specifically for heavily compressed social platforms like WhatsApp and Telegram. Instead of trusting raw RGB pixels alone—which fail under H.264 CRF 38 compression and miss real videos paired with fake captions—PixelProof combines fixed 5x5 Steganalysis Rich Model (SRM) high-pass visual filters, 16 kHz acoustic vocoder inspection, and deterministic CRC32 text-claim modeling through a Late-Fusion Cosine Discrepancy Head. Trained on 7,000 identity-disjoint FaceForensics++ C23 clips across 6 vision architectures on an RTX 4050 GPU, our calibrated Stacked Meta-Learner (`.pkl`) achieves **0.9674 test ROC-AUC (90.25% balanced accuracy and 95.30% authentic recall at threshold 0.61)**, while cross-modal Knowledge Distillation + QUInt8 ONNX quantization compresses the ResNet-50 teacher into a **0.92 MB** mobile model running at **0.16 ms** CPU latency."*

---

## 2. Core Concept & Motivation Questions

### Q1. Why did you choose this problem instead of standard image deepfake detection?
**Answer:** Standard deepfake detectors fail in two real-world scenarios: (1) dark-social platforms like WhatsApp recompress videos down to ~480p, 15 FPS, and H.264 CRF 38 quantization, wiping out fine GAN skin textures; and (2) **"cheapfakes"**—real, unedited news videos shared with false viral election captions—fool visual-only CNNs completely because every pixel is authentic. PixelProof tackles compression robustness, multimodal cheapfakes, and strict mobile deployment (`<= 45 MB`, `< 300 ms`) together.

### Q2. What are the 3 output classes in PixelProof?
**Answer:**  
- **Class 0 — `Authentic`:** Clean real video + neutral/no deceptive claim + clean audio stream.  
- **Class 1 — `Manipulated Payload`:** Face-swapped, reenacted, or neural-rendered visual pixels (or synthetic voice spoofing).  
- **Class 2 — `Contextual Misinformation`:** Real video pixels paired with a sensational, high-urgency, or contradictory viral caption (cheapfake).

### Q3. Why is the WhatsApp degradation simulator important in your demo?
**Answer:** In `src/degradation_pipeline.py`, FFmpeg transcodes an input video on demand to **<= 480p height, 15 FPS, H.264 CRF 38, and 16 kbps mono AAC audio**. This proves live in front of the examiner whether the detector survives realistic social-media compression rather than relying on pristine benchmark quality.

---

## 3. Dataset, Sampling & Leakage Prevention Questions

### Q4. Which dataset did you train on, and what are its exact statistics?
**Answer:** **FaceForensics++ C23** (`7,000` MP4 videos, uniform `25.0 FPS`, mean payload `2.44 MB`, median `1.48 MB`):  
- **1,000 Authentic (`original`)** clips and **6,000 Manipulated** clips split equally across 6 forgery families (`Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`, `FaceShifter`, `DeepFakeDetection`, 1,000 each).

### Q5. How did you prevent data leakage when splitting the dataset?
**Answer:** FaceForensics++ reuses the same actor IDs across real clips (`006.mp4`) and forged pairs (`006_002.mp4`). If we split randomly by clip, the CNN memorizes an actor's face in training and cheats in testing. `src/inspect_dataset.py` parses `source_id` across all **1,028 unique actor identities** and enforces **strict identity-disjoint partitioning**:  
- **Train:** `4,692` (`67.0%` — `704` Real, `3,988` Fake)  
- **Validation:** `1,216` (`17.4%` — `147` Real, `1,069` Fake)  
- **Held-Out Test:** `1,092` indexed (`15.6%` — `1,084` resolvable files on disk: `149` Real, `935` Fake, because `8` `FaceSwap` paths do not exist in the local archive).

### Q6. FaceForensics++ has a 6:1 Fake-to-Real imbalance (6,000 Fake vs 1,000 Real). How did you stop the model from always predicting "Deepfake"?
**Answer:** Three defenses worked together:  
1. **`WeightedRandomSampler`** in `src/train.py`, rebalancing mini-batches to **~42% Real / 58% Fake**.  
2. **Class-weighted Cross-Entropy** `[1.15, 0.90, 1.80]` with `0.04` label smoothing in `src/train_zoo.py` (`verimedia` weighted Class 0 Authentic and Class 2 Misinformation higher than majority Class 1).  
3. **Validation-calibrated Platt scaling (`.pkl`) + threshold optimization** targeting **Balanced Accuracy** (`0.5 * (Real Recall + Fake Recall)`), which lifted Authentic Recall to **95.30%** at stacked threshold `0.61`.

### Q7. Why did a clean original video earlier get flagged as Manipulated before retraining, and how did `--domain_finetune` fix it?
**Answer:** Models trained solely on raw C23 frames can overfit to H.264 encoder headers or clean pixel sharpness, treating any unfamiliar compression or mobile camera noise as a deepfake. Re-training with `--epochs 6 --no_resume --domain_finetune` injected clip-consistent crop zoom (`0.86–0.97x`), downscale-upscale blur (`96–184 px`), `YCbCr 4:2:0` chroma subsampling, and JPEG quality compression (`q=32–86`), preserving **0.9023 ROC-AUC** even under synthetic compression tests.

---

## 4. Computer Vision, SRM Steganalysis & Preprocessing Questions

### Q8. What is SRM (`SRMConv2d`), and why keep its weights frozen (`requires_grad=False`)?
**Answer:** SRM stands for **Steganalysis Rich Model** (`src/srm_filter.py`). It applies **3 fixed 5x5 zero-sum high-pass kernels** (`K1` 1st-order edge `/4`, `K2` 2nd-order square residual `/12` with center `-12`, and `K3` Laplacian `/4`), clamped to `[-3.0, +3.0]`, and blended via `x + 0.3 * SRM(x)`. Keeping the weights fixed prevents backpropagation from overwriting them into semantic color filters—forcing the CNNs to inspect sub-pixel face-swap boundaries and stego noise residuals.

### Q9. Why crop faces (`224x224`) with Haar Cascades instead of passing full `1920x1080` frames into the neural network?
**Answer:** Face manipulation happens inside the facial region (~10–25% of frame pixels). Feeding full `1080p` frames dilutes gradients on irrelevant TV news banners or room backgrounds and explodes VRAM usage. Pre-caching `8` uniform `224x224` Haar crops (`frontalface_default` + `frontalface_alt2` fallback with `20%` margin) took `11.27 min` once for all `7,000` clips and cut GPU epoch time dramatically.

### Q10. Why does PixelProof sometimes return `Inconclusive (Insufficient Face Evidence)`?
**Answer:** If a video shows landscapes or blank screens without a detectable human face across sampled frames (below the minimum Haar box floor `max(24, min(110, 5% of min dim))`), running a face-deepfake classifier would output random noise. `src/pipeline.py` flags it cleanly as `Inconclusive` and clears dissenting model warnings instead of fabricating a fake verdict.

---

## 5. Multimodal Audio, NLP & Late-Fusion Architecture Questions

### Q11. Explain the formula inside your Late-Fusion Discrepancy Head.
**Answer:** In `src/fusion_model.py` (`MultimodalLateFusionHead`):  
1. Apply **`LayerNorm`** independently to `V_f` (`576d–1280d`), `A_f` (`768d`), and `T_f` (`768d`) so no single modality dominates magnitude.  
2. Project all three into a shared `L2`-normalized metric space (`128d` in `VeriMediaZooModel`) to obtain unit vectors `p_v, p_a, p_t`.  
3. Compute pairwise cosine **discrepancies**:  
   $$D(V,A) = 1 - p_v^\top p_a,\quad D(V,T) = 1 - p_v^\top p_t,\quad D(A,T) = 1 - p_a^\top p_t \quad \in [0, 2]$$  
4. Compute elementwise cross-interaction `(p_v * p_a) + (p_v * p_t) + (p_a * p_t)`, concatenate everything with `[v_norm, a_norm, t_norm, disc_scalars]`, and pass through a **3-layer residual MLP** to output the 3 class logits.

### Q12. Why did you choose Late Fusion instead of Early Fusion?
**Answer:** Audio waveforms (`16 kHz` temporal `1D` sequences), text token IDs (`128` discrete tokens), and `2D` face crops (`N x 3 x 224 x 224`) live in completely incompatible spaces. Late fusion lets each branch extract specialized features cleanly, tolerates missing modalities (e.g., muted video or empty caption), and makes cross-modal discrepancy interpretable (`D(V,T)` score in the dashboard).

### Q13. How does your NLP branch work without downloading a huge 500 MB HuggingFace BERT/RoBERTa file?
**Answer:** In `src/nlp_branch.py`, `IntentClaimBackbone` uses a **stateless deterministic `zlib.crc32` hash tokenizer** (`vocab=10,000`, `max_len=128`, `[CLS]=1`, `[SEP]=2`) feeding a `256d` embedding + positional encoding + **2-layer Transformer Encoder** (`4` heads, `512` FFN) projected to `768d`, paired with heuristic misinformation lexicon checks (`urgency`, `conspiracy`, `fabricated_electoral`, exclamation density). Note why CRC32 matters: Python's built-in `hash()` changes across restarts (`PYTHONHASHSEED`), whereas CRC32 is 100% deterministic across Windows/Linux/ONNX!

### Q14. What bugs did you discover and fix in Audio extraction and Verdict conflict logic?
**Answer:** Two major real-world bugs were fixed:  
1. **Audio filename collision & silent false spikes (`src/audio_branch.py`):** Different videos sharing stems like `006.mp4` overwrote each other's WAV cache—fixed by appending an 8-char `MD5` hash of the full resolved path, clearing stale `.wav` files, and checking `res.returncode == 0`. Furthermore, silent videos (`RMS < 1e-3`) had random quantization floor phase jitter mimicking a vocoder—now guarded to return `0.0`.  
2. **Clickbait override & pure-visual stacker decoupling (`src/pipeline.py`):** Previously, putting a sensational caption on a *deepfake* video could let Class 2 outvote visual tampering—while conversely, claim-shifted softmax residuals inflated `raw_p_fake` and fooled the binary stacker on cheapfakes. Now we evaluate a pure visual-audio (`empty_tokens`) pass for the binary stacker so real clips + deceptive text accurately score **Contextual Misinformation (`88.27%` clean / `84.94%` WhatsApp CRF 38, `6/6` agree)**, while strong visual tampering (`p_fake_stack >= 0.70`) still overrides clickbait text cleanly (`74.51%`, `4/6`).

---

## 6. Model Zoo, Stacking Meta-Learner & Evaluation Metrics Questions

### Q15. Why train 6 separate architectures (`ResNet-50`, `EfficientNetV2`, `XceptionNet`, `MobileNetV3`, `Temporal-LSTM`, `MesoInception-4`) instead of just ResNet-50?
**Answer:** Each architecture has an inductive bias for different forgery artifacts:  
- **`ResNet-50` (`0.9516` AUC, stacker wt `3.7785`):** Deep residual anchor capturing multi-scale spatial manipulation.  
- **`EfficientNetV2-S` (`0.9408` AUC, stacker wt `3.1870`):** Strongest single backbone under compressed domain evaluation (`0.8687` compressed AUC).  
- **`XceptionNet` (`0.9203` AUC, stacker wt `1.8467`):** Depthwise separable convolutions excel at face-swap boundary seams.  
- **`MobileNetV3-Edge` (`0.9046` AUC, wt `1.4993`):** Fast baseline aligned with the edge student.  
- **`Temporal-LSTM` (`0.8958` AUC, wt `0.4892`):** 2-layer BiLSTM detects blinking/flickering dynamics across frames.  
- **`MesoInception-4` (`0.6410` AUC, wt `1.0700`):** Dilated mesoscopic filters focus strictly on `8x8/16x16` block artifacts.  
Combining all 6 via the balanced Logistic Regression Stacker boosts held-out test ROC-AUC to **`0.9674`** (clean) and **`0.9023`** (domain compressed), beating every single backbone.

### Q16. Why did `MesoInception-4` get lower accuracy (`0.6410` AUC), and why keep it in the committee?
**Answer:** `MesoInception-4` is an intentionally tiny, shallow CNN trained from scratch (`pretrained=False`, only `11.92M` total wrapper params) focused purely on low-level macroblock noise without deep semantic capacity. Even though weak standalone, its uncorrelated low-level error signal actually contributes useful diversity (`coef = 1.0700`) inside the meta-learner.

### Q17. Why do you emphasize ROC-AUC and Balanced Accuracy over raw Accuracy?
**Answer:** In our test set (`149` Real vs `935` Fake = `86.25%` Fake), a dumb classifier that blindly labels *every* video as "Fake" would score **86.25% raw accuracy** yet have **0% Real Recall** and fail completely in deployment! **Balanced Accuracy** `(Real Recall + Fake Recall) / 2` and **ROC-AUC** weight both classes equally. At `0.61`, our Stacked Committee reaches **90.27% Balanced Accuracy** and **95.30% Real Recall** (`142/149` authentic clips cleared, only `7` false alarms).

### Q18. Explain Platt Calibration and how `verimedia_committee_ensemble.pkl` works.
**Answer:** Deep networks often output overconfident probabilities pushed near `0.01` or `0.99`. On clean + domain-compressed validation outputs, **Platt scaling** fits a 1D `LogisticRegression(class_weight="balanced")` per model (`_calibrator.pkl`) to recalibrate probabilities, and the **Stacked Committee Meta-Learner** (`verimedia_committee_ensemble.pkl`) takes the 6 raw model `P(Fake)` scores as a 6-dimensional feature vector and learns optimal weights (`intercept = -6.2811`) plus the optimal threshold (`0.61`).

---

## 7. Knowledge Distillation, ONNX & Edge Deployment Questions

### Q19. Write and explain your Knowledge Distillation loss function.
**Answer:** From `src/distillation.py` (`alpha = 0.5`, `T = 2.0`, `discrepancy_weight = 0.2`):  
$$\mathcal{L}_{KD} = (1 - \alpha)\,\text{CE}(z_s, y) + \alpha\,T^2\,\text{KL}\!\left(\sigma(z_s / T)\,\|\,\sigma(z_t / T)\right) + 0.2\,\text{MSE}(D_s, D_t)$$  
- Hard `CE` learns true ground labels.  
- Temperature-scaled `KL` (`T=2.0`) transfers soft inter-class certainty ("dark knowledge") from `ResNet-50` Teacher (`144.21 MB`) to `MobileNetV3` Student (`49.23 MB` `.pth`).  
- `MSE(D_s, D_t)` forces the tiny student's visual-audio-text contradiction distances to mimic the large teacher.

### Q20. How did you shrink a 144.21 MB PyTorch model down to 0.92 MB?
**Answer:** Two steps: (1) **Knowledge Distillation** into `MobileNetV3` (`3.54 MB` when fusion head is exported to FP32 ONNX Opset 18), followed by (2) **8-bit Dynamic Post-Training Quantization (`QUInt8`)** via `onnxruntime.quantization.quantize_dynamic` (`3.83x` shrink -> **`0.92 MB`**), achieving **`0.16 ms` mean CPU latency** (`0.24 ms` p95) and **`0.46 MB` active RAM delta** over 50 benchmark runs—crushing the `<= 45 MB`, `< 300 ms`, `< 180 MB` System Verification Matrix limits.

### Q21. Why use Dynamic PTQ (`QUInt8`) instead of Static Quantization or QAT?
**Answer:** For edge MLP / Transformer-style projection heads (`VeriMediaEdgeONNXWrapper`), weights dominate memory and matrices multiply activations dynamically at runtime without needing calibration histograms or fragile GPU QAT graphs—delivering a `3.83x` size reduction cleanly on `CPUExecutionProvider`. (Good future work is quantizing the full convolutional backbone cleanly too!)

---

## 8. Engineering, Honest Limitations & "Trap" Questions Teachers Love

### Q22. Why did PyTorch initially fail on your RTX 4050 GPU and how did you fix it?
**Answer:** The environment initially installed PyTorch `cu126`, whose compiled CUDA kernels lacked native compatibility on this RTX 4050 (`Compute 8.9 sm_89`) paired with the host CUDA `13.3` driver (`cudaErrorInvalidKernelImage`). Upgrading `.venv` to **`PyTorch 2.14.1+cu130` (`CUDA 13.0`)** resolved kernel execution cleanly, letting all `6` models + distillation finish in `185.18 min`.

### Q23. Which deepfake category was hardest for your model, and why?
**Answer:** **`NeuralTextures`** (`66.44%` Stacked raw `@0.50`, `58.39%` `@0.61`, while `ResNet-50` alone reached `77.85%`), compared to `DeepFakeDetection` (`100.0%`), `Deepfakes` (`91.95%`), `Face2Face` (`91.95%`), and `FaceSwap` (`90.78%`). *Why?* Unlike full-face swaps that leave oval boundary seams around the forehead and jaw, NeuralTextures only modifies subtle mouth-interior pixels—making zoomed lip-ROI attention a clear next research upgrade.

### Q24. Be honest: What are the limitations of your project?
**Answer (Teachers reward honesty!):**  
1. **FaceForensics++ C23 contains muted/silent audio tracks (`had_audio=False`)**, so neural voice-cloning spoofing weights need end-to-end co-training on an audio-visual deepfake dataset like **FakeAVCeleb** or **ASVspoof** alongside our working DSP vocoder analyzer.  
2. **`NeuralTextures` subtle mouth-only edits** remain harder (`66.44%` stacked) than full face-swaps (`91.95%–100%`).  
3. **Out-of-distribution smartphone HDR videos** (vertical `60 FPS` iPhone `.MOV` sharpening) differ statistically from YouTube `25 FPS` FF++ clips and benefit from wild smartphone UGC calibration.  
4. **In-repo ONNX export scope:** `verimedia_edge_int8.onnx` quantizes the Multimodal Late-Fusion Head (`VeriMediaEdgeONNXWrapper`), while backbone feature extraction runs via `StudentVisualBackbone` in `VeriMediaEngine(use_onnx_edge=True)`.

### Q25. Why aren't the `.pth` checkpoints uploaded to your GitHub repo (`goyaljatin870/PixelProof`)?
**Answer:** Combined `.pth` checkpoints total ~**`560 MB`** (`ResNet-50 144.2 MB`, `Xception 133.9 MB`, `EfficientNetV2 124.5 MB`, etc.), exceeding GitHub's `100 MB` per-file limit. So `.gitignore` excludes heavy `.pth` files and raw media while keeping all source code, ONNX exports (`0.92 MB` INT8 + `3.54 MB` FP32), `.pkl` calibrators/stacker, reports, and `13/13` passing tests tracked cleanly—with automatic fallback to Standalone Forensic Mode if `.pth` files are absent.

---

## 9. Last-Minute Rapid-Fire Cheat Sheet (Read 2 Mins Before Viva)

| Question / Topic | Exact Verified Number / Answer |
| :--- | :--- |
| **Dataset & Actors** | FaceForensics++ C23 — **`7,000` clips**, `7` classes (`1,000` each), **`1,028` unique actors** |
| **Data Split (Zero Leakage)** | **`4,692` Train / `1,216` Val / `1,092` Test** (`1,084` resolvable files scored on disk: `149 Real / 935 Fake`) |
| **Best Ensemble ROC-AUC** | **Stacked Committee (`.pkl`): `0.9674`** clean test AUC (`0.9023` domain-compressed AUC) |
| **Balanced Acc & Real Recall** | **`90.25%`** Balanced Acc `@0.50` • **`95.30%` Real Recall `@0.61`** (`142/149` real right, only `7` FP) |
| **Best Single Backbone** | **`ResNet-50`: `0.9516` AUC** (`91.14%` raw acc); **`EfficientNetV2-S`: `0.9408` AUC** |
| **Highest Stacker Weights** | `ResNet-50` (`3.7785`), `EfficientNetV2` (`3.1870`), `Xception` (`1.8467`) |
| **Distilled Student Test AUC** | **`0.8935` ROC-AUC** (`82.84%` raw acc, `81.03%` balanced acc) |
| **Edge INT8 Verification** | **`0.92 MB`** (`<= 45 MB`), **`0.16 ms` mean / `0.24 ms` p95** (`< 300 ms`), **`0.46 MB` RAM delta** (`< 180 MB`) |
| **GPU & Runtime** | NVIDIA **RTX 4050 (`6 GB VRAM`, `sm_89`)**, PyTorch **`2.14.1+cu130`**, **`185.18 min`**, `6` epochs |
| **Unit Tests** | **`13 / 13` PyTest cases PASS** across SRM, branches, splits & ONNX INT8 |
