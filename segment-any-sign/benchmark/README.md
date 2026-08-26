# Benchmark

Inference-only evaluation: run existing models over the curated datasets, score
with [`../metrics/`](../metrics/), produce the table. No training here — that is a
future `experiments/`.

## Rules

**One evaluation protocol for every model, inherited from Moryossef & Jiang
(2023). Deviate only for an explicit bug, and record why.**

Same rule [`../metrics/`](../metrics/) follows for metric definitions. It fixes
the clip list, split, filters, gold, and frame conversion; a newer model is
measured against those whether or not it was built for them. The cost is intended
— the 2026 model gives up 0.13 phrase IoU this way — and buys a column that means
one thing all the way down.

A model's **preprocessing is not part of the protocol**: that is the model, not
the measurement. Forcing one model's onto another cost the 2023 model ~0.07 frame
F1.

| deviation taken | why it is a bug, not a preference |
|---|---|
| 3 of 17 test clips dropped | no annotation at all — a free ~1.0 on every metric, separating no models |
| upstream `segment_f1` dropped | computed `(p*r)/(p+r)`, half an F1; `mF1S` replaces it |

## Running it

```bash
conda activate sas2023                                    # the 2023 model only
python benchmark/predict_dgs_2023.py --split test --model model_E4s-1.pth

conda activate sas                                        # the 2026 model + scoring
python benchmark/predict_dgs_2026.py --split test
python benchmark/score.py benchmark/predictions/*.json
```

Clips, filters and gold come from
[`../datasets/public_dgs_corpus/load.py`](../datasets/public_dgs_corpus/load.py)
for every model: `iter_clips` reads the TFDS build at 25fps, `iter_clips_native`
the archived `.pose` originals at 50fps. Each model gets the one it was developed
against. Predictions land in `predictions/*.json` as segments plus RLE'd frame
labels.

## Results

Public DGS Corpus test — 9 documents, **14 annotated videos** of 17. Sign and
phrase as column groups, after Table 2 of the 2023 paper. Above the divider:
transcribed from publications. Below: ours.

`F1-ma`/`F1-mi` are frame-level F1 over O/B/I, macro/micro; `IoU` is pooled frame
overlap; `%` is `#pred / #gold`, optimal at **1** either way; `mF1S` is
matched-segment F1 ([`../metrics/`](../metrics/)). Models carry decoding
thresholds as `(sign b/o, phrase b/o)`.

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
| 2023 E1s (60/50, 90/90) | ours | 0.560 | 0.701 | 0.621 | 1.031 | 0.441 | 0.589 | 0.854 | 0.814 | 0.964 | 0.361 |
| 2023 E4s (60/50, 80/80) | ours | 0.553 | 0.695 | 0.620 | 1.074 | 0.429 | 0.593 | 0.858 | 0.817 | 1.073 | 0.376 |
| 2026 (argmax, 50fps) | ours | 0.530 | 0.800 | 0.611 | 0.941 | 0.436 | 0.513 | 0.861 | 0.792 | 0.437 | 0.071 |

**Our rows are deliberately not comparable to the published ones**, both
deviations pulling the same way: the 2023 rows lose the three free clips the paper
counted, the 2026 row is scored on 2023's phrase definition. The published figures
are the flattered ones.

`*` marks the 2023 paper's tuned decoding, which moves only IoU and `%` — hence
`—` for the decoding-independent F1. `F1-mi` and `mF1S` are unpublished by anyone.

**Hands-On** is the only follow-up on this corpus (others use BOBSL, YouTube-ASL,
BSL Corpus, ASLLRP, LSF, LSM). Indicative only: sign level, no thresholds stated,
split described just as the MeineDGS protocol.
See [`../literature/2025-hands-on/`](../literature/2025-hands-on/).

## Protocol

**The three dropped clips.** `1180022_b`, `1187154_a`, `1419122_a` are the
unannotated partner in a two-signer conversation — those documents carry
`Deutsche_Übersetzung` and `Lexem_Gebärde` tiers for one participant only. A
person is on camera but is the listener, and every model correctly predicts almost
nothing. With no gold and no prediction each scores ~1.0 throughout: sign IoU
0.679 across 17 against 0.611 across 14. `score.py --all-clips` restores the
17-video set. Upstream's 2026 eval independently drops the same three.

**What counts as a phrase.** The two codebases disagree, and neither definition is
ours:

| | phrase = | source |
|---|---|---|
| v2023 | first gloss → last gloss of a sentence | `data.py::build_classes_vectors` |
| 2026 | the `Deutsche_Übersetzung` tier's timeslots | `datasets/dgs/dataset.py:127` |

Sentence bounds are wider; the gloss extent sits inside them, trimming lead-in and
trail-out. Phrase IoU against both, same predictions:

| Model | vs gloss extent (**benchmark**) | vs sentence bounds |
|---|---|---|
| 2023 E1s | 0.814 | 0.863 |
| 2023 E4s | 0.817 | 0.849 |
| 2026 | 0.792 | 0.922 |

Every row uses the **2023 gloss extent**. Letting each model use its own
definition would measure the 2026 row against a target 0.13 IoU easier than the
2023 rows face. `predict_dgs_2026.py --phrase sentence` scores it on its own
terms. A differing definition is a design choice, not a bug — but standardising on
sentence bounds would raise every model rather than lower one, and they are the
human annotation rather than a derived quantity, so it is worth revisiting
deliberately.

**Decoding thresholds.** 2023 defaults are its grid-search values
(`v2023 src/summary_decoding_E4s.csv`, 82 configs, selected on dev): **sign b=60
o=50**, **phrase b=80 o=80**. IoU saturates across many configs, so `%` separates
them. The shipped v2023 CLI hardcodes phrase 90/90 regardless of checkpoint — that
is the *E1s* tuning; pass `--phrase-b 90 --phrase-o 90` for E1s. The 2026 model
uses argmax and has no thresholds; its README rejects threshold decoding.

## Reproductions

### Moryossef & Jiang (2023)

`--all-clips` figures, since the paper scored all 17:

| model | level | frame F1 | (paper) | IoU | (paper) |
|---|---|---|---|---|---|
| E1s | sign | 0.6378 | 0.63 | 0.6878 | 0.69 |
| E1s | phrase | 0.6615 | 0.65 | 0.8471 | 0.85 |
| E4s | sign | 0.5924 | 0.59 | 0.6280 | 0.63 |
| E4s | phrase | 0.6258 | 0.62 | 0.7902 | 0.79 |

IoU is compared against the tuned-decoding rows (E1s\*/E4s\*), which is what we
run; frame F1 is decoding-independent. `%` is not compared — the paper's come from
`likeliest` decoding, ours from thresholds.

`predict_dgs_2023.py` **drives the original v2023 code**. A from-scratch version
reproduced the segment metrics but sat ~0.07 low on the frame metrics, for reasons
invisible from the paper:

- `tfds_dataset.py` imports **its own `pose_utils`**, not pose-format's — its
  `pose_hide_legs` zeroes eight leg points *and* their confidences. The
  same-named pose-format helper costs ~0.07 frame F1.
- **`pose-format` version matters**: v2023 pinned `>=0.3.2`, 0.9.0 changes the
  scores. `sas2023` pins 0.3.2.
- Two different golds: `floor(t * fps)` inclusive for IoU and `%`, `build_bio`'s
  walk (effectively `ceil`) for frame metrics.
- **Frame F1 compares argmax of raw probabilities against `build_bio` labels**,
  never decoded segments.
- Its macro F1 passes **no label set**, averaging only over present classes;
  `score.py` passes `labels=None` to match.
- `<cmdp:Task>Joke</cmdp:Task>` documents are dropped, taking the test split from
  10 documents to 9.

### The 2026 model

Reproduces under *its* protocol: `--phrase sentence` gives phrase IoU 0.922
against a published 0.925, same 14 clips. Sign is 0.611 against 0.652 either way
(the sign definition never changed); the likeliest remaining difference is pose
provenance — upstream reads a `poses_dir` of MediaPipe Holistic poses, we read the
archived `.pose` downloads, and sign boundaries are the more extraction-sensitive
level. Unconfirmed.

Metrics were checked against their `evaluate.py`: macro frame F1 with no label
set, `segment_IoU` identical to our `global_iou`, argmax decoding, gold from BIO
labels, mean over clips. Three corrections were needed, each a trap for the next
model:

1. **It does not use TFDS** — its loader reads raw `.pose` and `.eaf` directly.
2. **fps.** The TFDS build baked in a 25fps downsample; the model publishes at 50.
   The 50fps originals were already in the download archive, keyed by
   `original_fname` in each `.INFO` sidecar — check there before rebuilding a
   config at ~144 GB and hours. Worth +0.06 sign IoU.
3. **Phrase definition**, above.

## Finding: IoU hides over-merging

The 2026 work selects on IoU (harmonic mean of sign and phrase); `%` and `mF1S`
appear nowhere in it. On its own phrase target it scores IoU 0.922 while `%` is
0.437 — covering nearly the right frames with **under half** the segments, merging
adjacent sentences into long runs. IoU is blind to this by construction: one
prediction spanning two gold phrases scores as well as two correct ones. `mF1S`
0.071 against the 2023 model's 0.376 is the same fact from the segment side.

Not a discrepancy with their numbers — a property their metrics cannot surface,
and the argument for reporting all four. Open: whether the merging is a decoding
artefact (argmax cannot force a boundary where the B posterior is weak but
non-zero) or learned, and whether a B-aware decode recovers `%` without giving up
IoU.

## Scope

- **Models:** 2023 (E1s / E4s) and 2026, with room for more.
- **Datasets:** whatever [`../datasets/`](../datasets/) has gold segments for.
- **Levels:** sign and phrase separate; an unannotated level renders blank, not
  zero.

Preprocessing differs genuinely: 2023 wants 3 components at 25fps, legs zeroed by
its vendored `pose_utils`, optionally optical flow and hand normalisation; 2026
wants `reduce_holistic` to 50 joints, mean/std normalisation from
`pose-anonymization`, velocity for 6 dims per joint. No shared "standard" pipeline
exists, and inventing one would break both.
