# Benchmark

Inference-only evaluation: run existing models over the curated datasets, score
with [`../metrics/`](../metrics/), and produce the benchmark table.

**No training or finetuning happens here.** That is reserved for a future
`experiments/`, so the table stays cheap to regenerate whenever a dataset or
model changes, with no training state involved.

## Running it

Two stages. Inference is expensive and environment-specific; scoring is neither,
so changing a metric never means re-running a model.

```bash
conda activate sas2023                                    # the 2023 model only
python benchmark/predict_dgs_2023.py --split test --model model_E4s-1.pth

conda activate sas                                        # the 2026 model + scoring
python benchmark/predict_dgs_2026.py --split test
python benchmark/score.py benchmark/predictions/*.json
```

`sas` runs the latest model and all scoring. Only the 2023 model needs an env of
its own, because its pose-format 0.3.2 pin cannot coexist with the >=0.8.1 the
2026 model requires.

**Data loading is shared, preprocessing is not.** Every model reads its clips,
filters and gold annotations from
[`../datasets/public_dgs_corpus/load.py`](../datasets/public_dgs_corpus/load.py),
so the table compares models rather than pipelines. What each model does to a pose
*after* loading is its own — and has to be, since substituting one model's
preprocessing for another's costs real points (see below). Each `predict_*.py`
owns that step.

Predictions land in `benchmark/predictions/*.json` — gold and predicted segments
plus run-length-encoded frame labels, a few hundred KB per run.

## Results

Public DGS Corpus, test split (9 documents / 17 videos). Laid out like Table 2 of
the 2023 paper: sign and phrase side by side as column groups rather than
stacked.

Rows above the *our benchmark results below* divider are transcribed from
published papers; rows under it are our own runs. Published rows are copied
verbatim, including the cells their authors left empty — nothing is recomputed or
back-filled.

Columns: `F1-ma` and `F1-mi` are frame-level F1 over O/B/I, macro- and
micro-averaged (see [`../metrics/`](../metrics/)); `IoU` is pooled frame overlap;
`%` is `#pred / #gold`, optimal at **1**, better or worse in either direction;
`mF1S` is matched-segment F1. Each model carries its decoding thresholds as
`(sign b/o, phrase b/o)`.

| | | Sign | | | | | Phrase | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Model** | **Source** | **F1-ma** | **F1-mi** | **IoU** | **%** | **mF1S** | **F1-ma** | **F1-mi** | **IoU** | **%** | **mF1S** |
| M&J 2023 E1s (50/50, 50/50) | paper, Tab. 2 | 0.63 | — | 0.69 | 1.11 | — | 0.65 | — | 0.82 | 1.63 | — |
| M&J 2023 E4s (50/50, 50/50) | paper, Tab. 2 | 0.59 | — | 0.63 | 1.13 | — | 0.62 | — | 0.79 | 1.43 | — |
| M&J 2023 E1s\* (60/50, 90/90) | paper, Tab. 2 | — | — | 0.69 | 1.03 | — | — | — | 0.85 | 1.02 | — |
| M&J 2023 E4s\* (60/50, 80/80) | paper, Tab. 2 | — | — | 0.63 | 1.06 | — | — | — | 0.79 | 1.12 | — |
| Hands-On 2025 (n/r) | paper, Tab. II | 0.86 | — | 0.76 | 0.98 | — | — | — | — | — | — |
| 2026 E169, 50fps | README | — | — | 0.652 | — | — | — | — | 0.925 | — | — |
| *— our benchmark results below —* | | | | | | | | | | | |
| 2023 E1s (60/50, 90/90) | ours | 0.638 | 0.754 | 0.688 | 1.026 | 0.441 | 0.662 | 0.880 | 0.847 | 0.971 | 0.361 |
| 2023 E4s (60/50, 80/80) | ours | 0.592 | 0.749 | 0.628 | 1.061 | 0.429 | 0.626 | 0.883 | 0.790 | 1.060 | 0.376 |
| 2026, 25fps (argmax) | ours | 0.589 | 0.823 | 0.619 | 0.884 | 0.424 | 0.568 | 0.881 | 0.770 | 0.539 | 0.080 |

Regenerate our rows with:

```bash
conda activate sas && python benchmark/score.py benchmark/predictions/*.json
```

Published rows are maintained by hand — `score.py` prints only what we ran.

**Reading the published rows.** `*` marks the 2023 paper's tuned decoding, which
changes only IoU and `%`; the paper prints `—` for F1 there because F1 is
decoding-independent, so the starred rows share the F1 of the rows above them.
Our two rows use the tuned thresholds, making **E1s\*/E4s\* the like-for-like
comparison** — and they match. No published work reports `F1-mi` or `mF1S` on
this dataset, so those columns are ours alone.

**The 2026 model is not yet reproduced, and its row is at the wrong fps.** Its
published numbers are 50fps; the only DGS config built on this cluster is
`holistic-25`, so our row is the model's **25fps operating point**. Sign IoU
0.619 against a published 0.652 is roughly what the fps gap predicts, but phrase
is not: 0.770 against 0.925, with `%` at 0.539 — it emits **half** the phrase
segments it should, merging adjacent sentences into long runs, which is also why
its phrase mF1S (0.080) is far below the 2023 model's. Treat this row as a first
working run, not a verified reproduction. Two things to rule out, in order:

1. **fps.** Building a `holistic-50` config settles it. The download cache
   (714 `.pose` files, 192 GB) looks complete, so no network is needed, but the
   build would add ~144 GB and run for hours — worth confirming before starting.
2. **Phrase decoding.** Argmax is what the 2026 README prescribes, and it reports
   phrase IoU 0.925 with it, so a plain merge at 25fps is the likelier story —
   but this is the next thing to check if fps does not explain it.

**Hands-On (n/r).** [Low et al. (2025)](https://arxiv.org/abs/2504.08593),
Table II, is the only follow-up we found that evaluates on the Public DGS Corpus;
the other follow-ups use BOBSL, YouTube-ASL, BSL Corpus, ASLLRP, LSF or LSM. Its
sign-level 0.86 F1 is far above everything else here, so treat the comparison as
indicative rather than settled: it reports **sign level only**, states no decoding
thresholds (hence `n/r`), and describes its split only as following the MeineDGS
translation protocol — we have not confirmed it is the same 17 videos. Its own
two baseline rows reproduce the 2023 E1s/E4s numbers exactly, which at least
fixes it to the same reference points. Unread as yet; see
[`../literature/2025-hands-on/`](../literature/2025-hands-on/).

`predict_dgs_2023.py` **drives the original v2023 code** rather than
reimplementing it — the source is vendored out of git into `.cache/v2023_src/`
and called directly. A from-scratch reimplementation reproduced the segment
metrics but sat ~0.07 low on the frame metrics, for reasons that are invisible
from a reading of the paper:

- `tfds_dataset.py` imports **its own `pose_utils`**, not pose-format's.
  Its `pose_hide_legs` zeroes exactly eight leg points *and* their confidences;
  substituting pose-format's same-named helper shifts the model input enough to
  cost ~0.07 frame F1.
- **`pose-format` version matters.** v2023 pinned `>=0.3.2`; running 0.9.0
  changed the scores. `sas2023` pins 0.3.2.
- The evaluation uses **two different golds**: `floor(t * fps)` spans, both ends
  inclusive, for IoU and percentage; `build_bio`'s walk (effectively `ceil`) for
  the frame metrics.
- **Frame F1 compares argmax of the raw probabilities against `build_bio`
  labels** — never the decoded segments.
- Its macro F1 passes **no label set**, so sklearn averages only over classes
  present. Three test clips have no annotation; scoring them 1.0 rather than
  0.33 moves the corpus mean by 0.12. `score.py` passes `labels=None` to match,
  and keeps those clips rather than dropping them.
- Documents tagged `<cmdp:Task>Joke</cmdp:Task>` are dropped, which is what takes
  the test split from 10 documents to 9.

## Decoding thresholds

Defaults are the tuned values from the 2023 grid search
(`v2023 src/summary_decoding_E4s.csv`, 82 configurations, selected on dev):
**sign b=60 o=50**, **phrase b=80 o=80**. IoU saturates across many
configurations, so the percentage metric is what separates them.

The shipped v2023 CLI hardcodes phrase 90/90 regardless of checkpoint — that is
the *E1s* tuning, and the grid rates it worse for E4s. Pass
`--phrase-b 90 --phrase-o 90` to reproduce E1s.

## Scope

- **Models:** the 2023 model (E1s / E4s) and the 2026 model, with room for more.
- **Datasets:** whatever [`../datasets/`](../datasets/) has gold segments for.
- **Levels:** sign and phrase kept separate; a dataset that annotates neither
  leaves that column blank rather than zero.

Each model owns its preprocessing, and the two are genuinely different: the 2023
model wants 3 components at 25fps, legs zeroed by its own vendored `pose_utils`,
optionally optical flow and hand normalisation; the 2026 model wants
`reduce_holistic` down to 50 joints, mean/std normalisation from
`pose-anonymization`, and velocity features appended for 6 dims per joint. There
is no shared "standard" pose pipeline, and inventing one would break both.

The gold annotations are shared, but the **frame conversion** is not: v2023 floors
both bounds of a span, while the 2026 code walks frame timestamps
(`create_bio_from_times`). Each is kept because each is what that model's
published numbers were computed with. The two differ by at most a frame — on the
test split, 0.3% of sign segments — so it does not carry the 2026 phrase gap.
