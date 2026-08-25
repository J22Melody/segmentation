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
conda activate sas2023                                    # the 2023 model's env
python benchmark/predict_dgs_2023.py --split test --model model_E4s-1.pth

conda activate sas                                        # no torch, no TF
python benchmark/score.py benchmark/predictions/*.json
```

Predictions land in `benchmark/predictions/*.json` — gold and predicted segments
plus run-length-encoded frame labels, a few hundred KB per run.

## Results

Public DGS Corpus, test split (9 documents / 17 videos). Laid out like Table 2 of
the 2023 paper: sign and phrase side by side as column groups rather than
stacked.

**Rows above the rule are transcribed from published papers; rows below are our
own runs.** Published rows are copied verbatim, including the cells their authors
left empty — nothing is recomputed or back-filled.

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
| | | | | | | | | | | | |
| **2023 E1s (60/50, 90/90)** | ours | **0.638** | 0.754 | **0.688** | **1.026** | **0.441** | **0.662** | 0.880 | **0.847** | **0.971** | **0.361** |
| **2023 E4s (60/50, 80/80)** | ours | **0.592** | 0.749 | **0.628** | **1.061** | **0.429** | **0.626** | 0.883 | **0.790** | **1.060** | **0.376** |

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

### Reproduction of Moryossef & Jiang (2023)

The DGS numbers are verified against the paper before anything else is added, so
that a later model's score means something:

| model | level | frame F1 | (paper) | IoU | (paper) |
|---|---|---|---|---|---|
| E1s | sign | 0.6378 | 0.63 | 0.6878 | 0.69 |
| E1s | phrase | 0.6615 | 0.65 | 0.8471 | 0.85 |
| E4s | sign | 0.5924 | 0.59 | 0.6280 | 0.63 |
| E4s | phrase | 0.6258 | 0.62 | 0.7902 | 0.79 |

IoU is compared against the *tuned-decoding* rows (E1s\*/E4s\*), since that is
what we run. Frame F1 is decoding-independent and so comes from the unstarred
rows. `%`, `F1-mi` and `mF1S` are not compared: the paper's percentages come from
its `likeliest` (argmax) decoding while ours use thresholds, and it publishes no
value at all for the other two.

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

- **Models:** the 2023 model (E1s / E4s); the 2026 model next, with room for more.
- **Datasets:** whatever [`../datasets/`](../datasets/) has gold segments for.
- **Levels:** sign and phrase kept separate; a dataset that annotates neither
  leaves that column blank rather than zero.

Each model owns its preprocessing. The 2023 model needs pose-format 0.3.2 and
the vendored `pose_utils`; the 2026 model will differ. That is why model
environments are separate, and why there is no shared "standard" pose pipeline.
