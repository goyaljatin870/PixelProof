# Dataset Directory Specifications

This directory holds dataset manifests and documentation on local dataset layout.

## 1. FaceForensics++ (FF++) Layout
FaceForensics++ is structured into original sequences and manipulated sequences across 3 compression levels:
- `c0` (raw / uncompressed)
- `c23` (light H.264 compression / high quality)
- `c40` (heavy H.264 compression / low quality)

### Expected Directory Hierarchy:
```text
data/FaceForensics++/
├── original_sequences/
│   └── youtube/
│       ├── c23/
│       │   └── videos/
│       │       ├── 000.mp4
│       │       └── ...
│       └── c40/
│           └── videos/
├── manipulated_sequences/
│   ├── Deepfakes/
│   │   ├── c23/
│   │   │   └── videos/
│   │   │       ├── 000_003.mp4
│   │   │       └── ...
│   │   └── c40/
│   ├── Face2Face/
│   │   └── c23/videos/
│   ├── FaceSwap/
│   │   └── c23/videos/
│   └── NeuralTextures/
│       └── c23/videos/
└── splits/
    ├── train.json
    ├── val.json
    └── test.json
```

### Video ID & Split Integrity:
- Original videos follow `XXX.mp4` (e.g. `000.mp4`).
- Manipulated videos follow `XXX_YYY.mp4` where `XXX` is the source actor video ID and `YYY` is the target actor video ID.
- The official split files (`train.json`, `val.json`, `test.json`) partition the 1,000 original IDs (e.g. 720 train / 140 val / 140 test).
- To prevent data leakage, all manipulated videos where `XXX` or `YYY` originates from a given split ID **must strictly remain in that same split**.

## 2. In-The-Wild Dataset (e.g. Celeb-DF v2) Layout
Celeb-DF v2 is a realistic in-the-wild deepfake benchmark with high visual quality and minimal boundary artifacts:
```text
data/in_the_wild/
├── Celeb-real/
│   └── *.mp4
├── Celeb-synthesis/
│   └── *.mp4
├── YouTube-real/
│   └── *.mp4
└── List_of_testing_videos.txt
```

## 3. Creating Manifests
Run the inspection script to automatically parse all directories, probe video headers (codec, resolution, fps, frame count), verify split IDs, and produce `manifest.csv` and `manifest_summary.json`:
```bash
python src/inspect_dataset.py \
    --ff_path "path/to/FaceForensics++" \
    --wild_path "path/to/in_the_wild" \
    --splits_path "path/to/FaceForensics++/splits" \
    --output_dir "data/manifests"
```
