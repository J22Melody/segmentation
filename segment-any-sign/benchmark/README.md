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

## Reproduction of Moryossef & Jiang (2023)

The DGS numbers are verified against the paper before anything else is added,
so a later model's score means something. Test split, 9 documents / 17 videos —
matching the paper's own count:

| model | level | frame F1 | (paper) | accuracy | (paper) | IoU | (paper) |
|---|---|---|---|---|---|---|---|
| E1s | sign | 0.6378 | 0.63 | 0.7540 | 0.75 | 0.6878 | 0.69 |
| E1s | phrase | 0.6615 | 0.65 | 0.8799 | 0.88 | 0.8471 | 0.82 |
| E4s | sign | 0.5924 | 0.59 | 0.7488 | 0.75 | 0.6280 | 0.63 |
| E4s | phrase | 0.6258 | 0.62 | 0.8832 | 0.88 | 0.7902 | 0.79 |

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
