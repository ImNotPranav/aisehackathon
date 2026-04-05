# PM2.5 Pollution Forecasting — IIT Delhi Hackathon Phase 2, Theme 2

Spatiotemporal deep learning solution for 16-hour ahead PM2.5 forecasting
across a 140×124 grid covering the Indian subcontinent.

**Current leaderboard score: 0.8517**

---

## Problem Statement

Given 10 hours of historical multi-feature meteorological and pollution inputs
at every grid point, predict PM2.5 surface concentration for the next 16 hours
across India at 25km × 25km resolution.

- **Input:** `(10, 140, 124, 12)` — lookback × height × width × features
- **Output:** `(16, 140, 124)` — horizon × height × width
- **Submission:** `(218, 140, 124, 16)` float32, no negatives, no NaNs

---

## Data

WRF-Chem simulation data, India 2016, hourly resolution.

| Split | Months | Hours |
|---|---|---|
| Train | April, July, October, December | 715 / 739 / 739 / 739 |
| Val | Last 150h of each month (rotating) | ~600 total |
| Test | `test_in/` pre-windowed | 218 samples |

### Features (12 total)

| Group | Features |
|---|---|
| Target | `cpm25` (PM2.5 µg/m³) — also used as input |
| Meteorological | `q2`, `t2`, `u10`, `v10`, `swdown`, `pblh`, `psfc`, `rain` |
| Engineered | `wind_speed`, `wind_dir`, `cpm25_roll3` |

**Notes:**
- Emission features (`PM25`, `NH3`, `SO2`, `NOx`, `NMVOC_e`, `NMVOC_finn`, `bio`) dropped — all ~1e-7 max, effectively zero noise
- `hour_sin`/`hour_cos` excluded — `time.npy` absent in `test_in/` causes train/test mismatch
- Normalization: robust IQR per feature per pixel, computed from training data only
- Do NOT use the provided `feat_min_max.mat` — statistics are unrepresentative

---

## Model Architecture

### Overview
```
Input (B, T=10, H=140, W=124, C=12)
│
▼
ConvLSTMEncoder         ← processes T steps sequentially, preserves temporal ordering
│
WindSpatialBias         ← sigmoid gate from last-step wind_speed + wind_dir
│
▼
U-Net Encoder           base=80 → 160 → 320 → 640
│
Bottleneck              640 → 1280  (CBAM attention)
│
U-Net Decoder           1280 → 640 → 320 → 160  (skip connections, CBAM at top)
│
Head                    Conv → GELU → Conv(16)   (zero-initialized)
│
Persistence Residual    output = α·delta + (1-α)·last_known_pm25
│
▼
Output (B, 16, H, W)
```

### Key Components

**ConvLSTMCell** — processes spatial feature maps through LSTM gates while
preserving 2D structure. Outperforms flat Conv3d temporal collapse for gridded
PM2.5 (confirmed by 3 independent papers).

**WindSpatialBias** — takes last-step `wind_speed` and `wind_dir` as a 2-channel
map, produces a learned sigmoid gate that multiplicatively modulates encoder
features. Structurally encodes wind transport direction.

**CBAM Attention** — channel + spatial attention at enc3, bottleneck, and dec3.

**Persistence Residual** — `persist_w` parameter (init=-2.0 → sigmoid≈0.12)
blends model delta with last known PM2.5 value. Model learns corrections to
persistence rather than absolute values.

**Parameters:** ~50M (base=80)

---

## Training

Optimizer : AdamW, lr=3e-4, weight_decay=1e-4
Scheduler : OneCycleLR, pct_start=0.1, cosine annealing, 70 epochs
Grad clip : 0.5
Batch size: 4
Stride    : 1 (dense sampling)
Norm      : GroupNorm throughout (BatchNorm caused NaN with ConvLSTM)
Precision : float32 only (autocast caused NaN with this architecture)

### Loss Function

total = 0.35 × GlobalSMAPE
+ 0.35 × EpisodeSMAPE   (episode σ=1.5, min 200 pixels)
+ 0.20 × EpisodePearson  (clamped [-1,1], valid mask guard)
+ 0.10 × HuberGradLoss   (spatial gradient, delta=1.0)

Episode pixels at timestep t: spatial points where `y > mean + 1.5σ`.
The loss specifically upweights high-pollution events which drive the
competition's EpisodeCorr and EpisodeSMAPE components.

### Validation Split

Last 150 hours of each of the 4 training months held out → rotating seasonal
validation. Prevents the checkpoint from overfitting to a single season's
pollution dynamics.

### Checkpoint Averaging

Top-3 checkpoints by val loss are averaged at the end of training, reducing
sensitivity to the final epoch.

---

## Scoring

The competition metric is a weighted average of three normalized components
(weights undisclosed):
NormGlobalSMAPE  = 1 - GlobalSMAPE / 2
NormEpisodeSMAPE = 1 - EpisodeSMAPE / 2
NormEpisodeCorr  = (EpisodeCorr + 1) / 2
Score = w1·NormGlobalSMAPE + w2·NormEpisodeCorr + w3·NormEpisodeSMAPE

Episode pixels per timestep: `y > mean(y_t) + 2.5σ(y_t)` over all spatial points.
Pearson correlation and SMAPE are computed only over these high-pollution pixels,
then averaged across the 16 forecast timesteps.

A local proxy scorer (`compute_score`) is included in the notebook using equal
weights as an approximation.

---

## Inference

**6-way Test Time Augmentation:**

| Pass | Transform |
|---|---|
| p1 | Original |
| p2 | Horizontal flip (exact inverse) |
| p3 | Vertical flip (exact inverse) |
| p4 | H+V flip (exact inverse) |
| p5 | + Gaussian noise (σ=0.02) |
| p6 | − Gaussian noise (σ=0.02) |

`Final = (p1 + p2 + p3 + p4 + p5 + p6) / 6`

Symmetric noise passes (p5, p6) cancel bias. Geometric flips are exact inverses.

Denormalization: `pred × cpm25_iqr + cpm25_med`, then clip to ≥0.

---

## Notebook Structure

| Cell | Description |
|---|---|
| 1 | Config — paths, hyperparameters, feature lists |
| 2 | Load training data, engineer features |
| 3 | Lock feature keys, compute IQR normalization stats |
| 4 | Dataset class with rotating seasonal val split |
| 5 | Model — ConvLSTMEncoder, WindSpatialBias, UNetPM25 |
| 6 | Loss functions — SMAPE, Pearson, Huber gradient |
| 7 | Training loop with NaN guards + checkpoint averaging |
| 7b | Validation scorer — exact competition metric on val set |
| 8 | Load and prepare test data |
| 9 | Build test tensors, sanity check |
| 10 | Inference with 6-way TTA |
| 11 | Save and validate submission |

---

## Hardware

- Platform: Kaggle (2× NVIDIA T4, 15GB VRAM each)
- RAM: 30GB
- Session limit: 12 hours
- Output storage: ~19.5GB

---

## What Was Tried / Key Decisions

| Decision | Reason |
|---|---|
| Emission features dropped | All ~1e-7 max → zero noise |
| IQR normalization | Robust to PM2.5 outliers vs min-max |
| ConvLSTMEncoder | Papers confirm outperforms flat Conv3d for gridded PM2.5 |
| WindSpatialBias | Wind direction defines PM2.5 transport paths |
| Persistence residual | PM2.5 changes slowly; learn delta not absolute |
| Episode σ=1.5 in loss | Catches moderate pollution events, not just extremes |
| GroupNorm throughout | BatchNorm caused NaN with ConvLSTM |
| Zero-init head | Stable training from epoch 1 |
| Huber spatial gradient loss | Preserves sharp plume boundaries |
| 6-way TTA with ±noise pair | Symmetric noise cancels bias |
| Top-3 checkpoint averaging | Reduces sensitivity to final epoch |
| hour_sin/hour_cos excluded | Bug fix: test has no time.npy → zeros → mismatch |
| No autocast | fp16 caused NaN with ConvLSTM architecture |
| Grad clip=0.5 | Prevents early gradient explosion |

---

## Potential Further Improvements

- Align training loss exactly to competition metric formula
- Stacked 2-layer ConvLSTM for deeper temporal modelling
- SimVP recurrence-free model as lightweight ensemble partner
- Separate models per season, ensemble at inference
- Curriculum learning: global loss first, episode loss phased in
- Longer lookback (>10h) if test format allows
- VMD decomposition of cpm25 input (frequency components as features)
- Transformer temporal attention block
- Larger base (>80) if VRAM permits

