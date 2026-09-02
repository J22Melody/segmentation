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

## Ablations

Filled in as runs land. Each row differs from the one above it by one component.

| # | experiment | change | sign IoU | phrase IoU | notes |
|---|---|---|---|---|---|
| 00 | 2023 reproduction | — | | | target 0.621 / 0.814 |
