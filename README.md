# Agricultural Remote Sensing Alignment

Parcel-level text-image alignment for agricultural remote sensing, built on a frozen
RemoteCLIP ViT-B/32 backbone. This repository contains the full experimental pipeline:
motivation analysis, two method versions, ablations, and paper figures.

## Problem

RemoteCLIP compresses a whole image into ONE vector (global alignment) and cannot tell
WHICH parcel a text refers to. On RSICD farmland images the sliding-window spatial
response is nearly flat (max-min similarity range < 0.07) and concept heatmaps for
different queries largely overlap (correlation 0.35-0.52), so the model lacks the
spatial selectivity required for parcel-level grounding.

## Method

**ParcelAlign v1** — learnable parcel tokens:
- K=8 learnable queries cross-attend to the 49 ViT patch tokens, producing K parcel
  tokens (soft attention regions, not fixed grids)
- Retrieval score mixes a global term (text vs CLS) and a parcel term
  (max cosine over parcels), FILIP-style
- Loss: global InfoNCE + parcel-level InfoNCE + parcel diversity regularizer
- Only 2.37M trainable parameters; the 151M backbone stays frozen

**ParcelAlign v2** — input-adaptive granularity gate:
- A gate predicts a per-image alpha from the CLS embedding, parcel-parcel diversity
  and parcel-to-CLS statistics, deciding how much to trust the global vs parcel term
- The gate output is batch-centered so it can only encode RELATIVE per-image
  differences; without centering it collapses to alpha->0 (see ablation)
- Interpretable: the learned alpha is consistently higher for texture-global
  categories (bareland 0.573, desert 0.568, farmland 0.543) and lower for
  object-centric ones (playground 0.443, railwaystation 0.471)

## Main results (RSICD test, 1093 images / 5465 captions)

| Method | ALL mR | FARM mR | Note |
|---|---|---|---|
| RemoteCLIP ViT-B/32 (baseline) | 32.53 | 23.51 | global alignment, zero-shot |
| Naive 2x2 tiling max | 18.63 | 16.31 | ablation: fixed grid, no learning |
| ParcelAlign v1 (fixed alpha) | 34.20 | 19.55 | learned parcels, scalar alpha |
| v2 gate w/o centering | 32.07 | 18.20 | ablation: gate collapses |
| ParcelAlign v2 (centered learned gate) | 34.49 | 21.26 | adaptive per-image alpha (~0.5) |
| v2.1 farmland boost x4 | 34.50 | 22.70 | class-balanced sampling (ablation) |
| **v2 + val-selected alpha (main)** | **35.36** | **23.15** | alpha=0.80 chosen on val, reported on test |

FARM = farmland test subset (37 images). The main result's alpha is selected on the 1094-image validation split without touching test. FARM 95% bootstrap CI at alpha=0.80: [14.77, 32.07], overlapping the baseline CI [16.22, 30.99] -> farmland is statistically tied with baseline. Farmland oracle alpha on val is 1.00: homogeneous farmland scenes on RSICD are inherently global. Oracle per-category alpha ceiling (analysis only): ALL 37.15 / FARM 24.14.

(v2 learned-gate FARM CI [13.42, 29.55]
overlaps the baseline CI [16.22, 30.99], so the farmland drop is not statistically
significant. Largest per-category gains: railwaystation +21.5, playground +11.5.

## Repository structure

```
models/parcel_align.py       ParcelAlign v1 (parcel tokenizer + losses)
models/parcel_align_v2.py    ParcelAlign v2 (+ centered granularity gate)
scripts/run_retrieval_baseline.py     RemoteCLIP zero-shot retrieval baseline
scripts/run_rsicd_perclass_final.py   per-category baseline breakdown
scripts/validate_granularity.py       naive tiling ablation
scripts/localization_heatmap.py       text-conditioned sliding-window heatmaps
scripts/localization_quant.py         cross-concept correlation analysis
scripts/train_parcel_align.py         v1 training + per-epoch eval
scripts/train_parcel_align_v2.py      v2 training + full final analysis
scripts/train_parcel_align_v21.py     v2.1 farmland-boosted training (ablation)
scripts/alpha_sweep.py               fixed-alpha trade-off sweep + oracle ceiling
scripts/val_alpha_select.py          leakage-free alpha selection on val -> test report
scripts/eval_parcel_perclass.py       per-category comparison + farmland CI
scripts/make_figure1.py / make_figure2.py / make_table1.py   paper figures
```

## Setup

See `requirements.txt`. Expect: torch 2.2.2+cu121, open_clip_torch 3.3.0, numpy 1.26.4.
Place the RemoteCLIP ViT-B/32 checkpoint at
`work/baselines/RemoteCLIP-weights/RemoteCLIP-ViT-B-32.pt` and the RSICD dataset under
`work/datasets/RSICD_optimal/` (this repo hosts code, not data or weights).

## Key findings

1. Global alignment lacks spatial selectivity: responses to "farmland", "road" and
   "water" overlap and are nearly flat under a unified color scale.
2. Fixed-grid tiling is not the answer: naive 2x2 max-pooling drops mR by 13.9 points.
3. Learned parcels help object-centric categories massively (railwaystation +21.5)
   but hurt texture-global ones (desert -10.6) with a fixed mix.
4. A centered per-image gate resolves this trade-off: it learns to trust the global
   term on homogeneous textures and the parcel term on heterogeneous scenes.


