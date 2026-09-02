# Experiments

Training and finetuning. [`../benchmark/`](../benchmark/) stays inference-only on
published checkpoints; anything with a training loop lives here.

**Goal: a controlled path from the 2023 model to the 2026 one.** Reproduce 2023
first, in the 2026 codebase and environment, then add one component at a time so
every gain has an owner. The published 2026 model changes the architecture, the
features, the augmentation, the loss and the decoding at once — its README lists
what helped, but not measured against a common baseline in one codebase.

## Rules

The [benchmarking rules](../benchmark/README.md#rules) apply unchanged: one
evaluation protocol for every run, inherited from 2023. An experiment may change
the *model*; it may not change how it is scored. Final numbers come from
`benchmark/score.py` on the same DGS test clips, so an experiment row and a
benchmark row are directly comparable.

Each run also reports **dev** numbers — test is for the final table only.

## Layout

```
experiments/
  README.md          this file — the plan and the ablation table
  <NN>_<slug>/       one directory per experiment
    README.md        what changed, why, and the result
    train.sh         the exact command (SLURM), so the run is repeatable
```

Numbered so the sequence is legible: `00_` is the 2023 reproduction, and each
later number adds exactly one component to the one before it.

## Step 0: reproduce 2023 in the 2026 code

Target: E1s at sign IoU 0.621 / phrase IoU 0.814 on our protocol
([benchmark table](../benchmark/README.md#results)) — trained from scratch here,
not the published checkpoint.

Two things stand in the way, both worth settling before any ablation:

1. **The 2026 code has no LSTM.** `args.py` exposes `hidden_dim`,
   `encoder_depth`, `attn_nhead` — the CNN-UNet + RoPE stack is hardcoded in
   `model/pose_encoder.py`. Reproducing E1s (BiLSTM, hidden 256, depth 4) means
   adding an encoder switch. Best contributed upstream rather than forked here.
2. **Training data layout.** `DGSSegmentationDataset` wants
   `<corpus>/videos/<doc>/data.eaf` plus poses keyed by video MD5; our archive is
   keyed `<doc>_<person>.pose`. The 2026 code has a `DATASET_REGISTRY`, so the
   clean route is registering an adapter over
   [`../datasets/public_dgs_corpus/load.py`](../datasets/public_dgs_corpus/load.py)
   rather than rebuilding the directory tree. That also keeps training and
   evaluation on one clip list.

Training also needs the `[train]` extras (wandb, optuna, lxml), which `sas` does
not install yet, and a GPU — so `train.sh` per run rather than ad-hoc commands.

## What differs, apart from the architecture

Read off both codebases (`v2023 src/` and `main sign_language_segmentation/`).
This is the candidate list to ablate — the architecture swap is deliberately not
in it.

### Data and features

| | 2023 | 2026 |
|---|---|---|
| source | TFDS `holistic-25` build | raw `.pose` + `.eaf`, read directly |
| fps | fixed 25 | native 50 |
| pose cleanup | own `pose_utils.pose_hide_legs` — zeroes 8 leg points **and their confidences** | `preprocess_pose`: `pose_hide_legs` → `reduce_holistic` → `normalize_mean_std` (pose-anonymization) |
| face | dropped | dropped |
| extra features | E4 only: optical flow + 3D hand normalisation | velocity (fps-normalised), always on |
| input | 3 components, xyz | 50 joints × 6 dims |

### Labels

| | 2023 | 2026 |
|---|---|---|
| BIO ids | `O=0, B=1, I=2` | `UNK=0, O=1, B=2, I=3` |
| span → frames | `build_bio` walk for training/frame metrics; `floor`/`floor` inclusive for segment metrics | `create_bio` (floor/ceil), or `create_bio_from_times` (searchsorted on timestamps) when `fps_aug` |
| phrase = | first gloss → last gloss | the `Deutsche_Übersetzung` tier's own bounds |
| clip set | keeps signer-videos with no glosses | drops them |
| split | TFDS split config | `splits.json`, extending `split.3.0.0-uzh-document` |

### Training

| | 2023 | 2026 |
|---|---|---|
| sampling | whole videos, no windowing | random 1024-frame windows |
| augmentation | **none** | `fps_aug` (25–50 random per clip, 5% tempo stretch), `frame_dropout` 0.15, `body_part_dropout` 0.1 |
| loss | NLL with **inverse class-frequency weights** per level | plain NLL + **Dice on the sign head** (weight 1.5) |
| loss masking | `(loss * mask).mean()` — scaled by mask density | `(loss * mask).sum() / mask.sum()` |
| optimiser | Adam, lr 1e-3, `ReduceLROnPlateau` | AdamW (wd 0.01), lr 5e-4, OneCycle |
| epochs / patience | 100 / 20 | 400–500 / 100 |

### Inference and scoring

| | 2023 | 2026 |
|---|---|---|
| decoding | thresholds `b`/`o`, tuned per level on dev | argmax (`likeliest`) |
| long clips | whole sequence in one pass | chunked at `num_frames` (1024) |
| selection metric | dev loss, early stopping | harmonic mean of sign and phrase IoU |
| reported | frame F1, IoU, % | IoU only |

Roughly ordered by expected effect from the 2026 README's own account: `fps_aug`
is called essential (0.58→0.49 without it), `frame_dropout` essential, Dice worth
+2pp sign IoU, velocity +1–2pp. Those are its numbers against its own baseline,
not ours — establishing them against a common baseline is the point of this
directory.

## Ablations

Filled in as runs land. Each row differs from the one above it by one component.

| # | experiment | change | sign IoU | phrase IoU | notes |
|---|---|---|---|---|---|
| 00 | 2023 reproduction | — | | | target 0.621 / 0.814 |
