# Slot-BERT Dataset Inspection

> These are **sample** clips (1 per dataset) shipped for demo. Full datasets are much larger.

---

## The 2nd Dimension — Frames, Not Seconds

The `T` dimension (29–30) represents **individual frames**, NOT seconds. Per the paper:
- Original videos are 30 seconds at 60 FPS
- Downsampled to **1 FPS** → 30 frames per clip (1 frame per second)
- So each frame = 1 second of real-time, but it's still a frame count
- Some samples have 29 frames (possibly off-by-one in clipping)
- During training, only **5 consecutive frames** are used per iteration (`Video_len=5`), with epoch-based sliding window

---

## Quick Reference

| Dataset | PKL Type | Shape (C,T,H,W) | Labels | Resolution | Frames | Full Dataset Size |
|---------|----------|------------------|--------|------------|--------|-------------------|
| **cholec80** | dict `{frames, labels}` | `(3, 29, 256, 256)` | `(29, 7)` binary tool presence | 256×256 | 29 | 6,300 clips (5,296 train) |
| **MICCAI** | raw ndarray | `(3, 29, 128, 128)` | External (`labels.csv`) | 128×128 | 29 | 24,642 clips (24,542 train) |
| **Thoracic** | dict `{frames, labels}` | `(3, 30, 256, 256)` | Empty `[]` | 256×256 | 30 | 264 clips |
| **MICCAI_selected_GT** | list of strings | N/A | N/A | N/A | N/A | 24,595 filenames |

All frame data: `uint8` (0–255), RGB, `(C, T, H, W)` layout.

---

## Sample Frames

### cholec80 — Cholecystectomy surgery (Grasper + Hook visible)
![cholec80 frame 0](sample_outputs/cholec80_frame0.png)

### MICCAI — Abdominal surgery (force bipolar + monopolar curved scissors + cadiere forceps)
![MICCAI frame 0](sample_outputs/miccai_frame0.png)

### Thoracic — Thoracic surgery (Robot-Assisted Thoracoscopic)
![Thoracic frame 0](sample_outputs/thoracic_frame0.png)

> Sample videos saved in `sample_outputs/`: `cholec80_clip_002742.mp4`, `miccai_clip_020283.mp4`, `thoracic_Kazu.mp4` (all at 5 FPS playback)

---

## 1. cholec80 — `output_pkl_croped/clip_002742.pkl`

- **Structure**: `dict` with keys `['frames', 'labels']`
- **frames**: `ndarray (3, 29, 256, 256)`, dtype=uint8, range [0, 255], mean≈93
- **labels**: `ndarray (29, 7)`, dtype=int64 — per-frame binary tool presence
- **The 7 columns** = 7 surgical tool categories:

| Col | Tool | Col | Tool |
|-----|------|-----|------|
| 0 | Grasper | 4 | Clipper |
| 1 | Bipolar | 5 | Irrigator |
| 2 | Hook | 6 | SpecimenBag |
| 3 | Scissors | | |

- This sample's labels: `[1, 0, 1, 0, 0, 0, 0]` across all 29 frames → Grasper + Hook present throughout
- **Sample clips provided**: 1 (full dataset: 6,300 clips)

## 2. MICCAI — `video_clips_pkl/clip_020283.pkl`

- **Structure**: Raw `ndarray` (NOT a dict — just video frames)
- **Shape**: `(3, 29, 128, 128)`, dtype=uint8, range [0, 255], mean≈85
- **Labels**: Not in pkl. From `MICCAI/labels.csv` (24,695 rows):
  - `clip_020283` → `[force bipolar, nan, monopolar curved scissors, cadiere forceps]`
  - 14 tool categories total (video-level, not per-frame)
- **Lower resolution** (128 vs 256) — pre-resized differently than cholec80
- **Sample clips provided**: 1 (full dataset: 24,642 clips)

## 3. Thoracic — `pkl/Kazu_RATS_RUL_#7_69_phase2_8.pkl`

- **Structure**: `dict` with keys `['frames', 'labels']`
- **frames**: `ndarray (3, 30, 256, 256)`, dtype=uint8, range [0, 255], mean≈80
- **labels**: Empty list `[]` — training data has no labels
- **5 categories** (eval only): Lymph node, Vagus nerve, Bronchus, Lung parenchyma, Instruments
- **30 frames** (full clip, no off-by-one)
- **Sample clips provided**: 1 (full dataset: 264 clips)

## 4. MICCAI_selected_GT — `unselected_videos.pkl`

- **Structure**: `list` of 24,595 strings (`'clip_019357.mp4'`, `'clip_001129.mp4'`, ...)
- **Purpose**: Train/eval split filter
- Code takes intersection of all MICCAI pkls with this list → **training set** (unannotated)
- Clips NOT in this list have GT annotations → **evaluation set**
- "unselected" = not selected for GT annotation = training pool

---

## Why Labels for Some, None for Others?

**Slot-BERT is self-supervised. Labels are NEVER used in training loss.**

Training loss:
1. **MSE feature reconstruction** — reconstruct frozen ViT (DINO) features from decoded slots
2. **Slot orthogonality loss** — contrastive, forces slots apart

Labels exist only for **post-hoc evaluation** (Jaccard, Dice, ARI, F1, etc.).

| Dataset | Labels during training | Labels during evaluation |
|---------|----------------------|------------------------|
| cholec80 | Passed to `optimization()` but **ignored** | Binary tool presence metrics |
| MICCAI | Loaded from CSV but **never in loss** | Tool detection metrics |
| Thoracic | Hardcoded to `np.ones(5)` — **dummy** | From annotated eval subset |

---

## Training Configuration (from paper + code)

| Parameter | Value |
|-----------|-------|
| Batch size | 1 (code) / 4 (paper) |
| Context window | 5 frames per iteration |
| Input resolution | Resized to 224×224 before ViT |
| Patch size | 16×16 |
| Num slots | 7 |
| Slot dim | 64 |
| Backbone | ViT-B/16 (DINO, frozen) |
| Optimizer | Adam, lr=1e-4, weight_decay=1e-5 |
| Grad clipping | 0.05 |
| Slot attention iters | 3 (first frame), 2 (subsequent) |

**Epochs**: MICCAI=80, Cholec=100, EndoVis/Thoracic=2000, Transfer=10-80

---

## Training Commands

```bash
# Train on MICCAI (default)
python main.py --mode train_miccai

# Train on Cholec80
python main.py --mode train_cholec

# Train on Thoracic
python main.py --mode train_thoracic

# Evaluate
python main.py --mode eval_cholec
python main.py --mode eval_thoracic
```

---

## Docker

```bash
# Build
docker compose build

# Train (default: cholec)
docker compose up

# Train specific dataset
docker compose run slot-bert python main.py --mode train_miccai

# Interactive shell
docker compose run slot-bert bash
```

---

## Pipeline Flow

```
Video PKL (C,T,H,W) → Crop 5 frames → Resize 224×224
    → Frozen ViT-B/16 (DINO) features
    → MLP encoder → Slot Attention (7 slots, 2-3 iters)
    → Temporal Slot Transformer (3 layers, 8 heads, 20% masking)
    → MLP decoder → Reconstruct ViT features
                     ↑
          Self-supervised loss (MSE + orthogonality)
```

Labels only enter at eval: "Do the 7 discovered slots correspond to the 7 real tools?"
