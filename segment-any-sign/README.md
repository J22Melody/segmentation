# segment-any-sign

Project workspace for *Segment Any Sign*, kept separate from the upstream
`sign-language-processing/segmentation` code so it does not collide with ongoing
development there.

Proposal by Zifan (@J22Melody), after discussion with Mathias (@bricksdont).

## Motivation

A continuation of the segmentation model from
[Moryossef & Jiang (2023)](https://aclanthology.org/2023.findings-emnlp.846/),
which has seen several follow-ups:

- [Revisiting subtitle-unit segmentation](https://aclanthology.org/2025.acl-srw.93/) — BIO tagging and optical flow in a Seq2Seq formulation, on BOBSL and YouTube-ASL.
- [Stronger hand-centric features](https://arxiv.org/abs/2504.08593) — HaMeR features and heavier Transformer backbones.
- [SAGE](https://openaccess.thecvf.com/content/ICCV2025W/MSLR/papers/Low_SAGE_Segment-Aware_Gloss-Free_Encoding_for_Token-Efficient_Sign_Language_Translation_ICCVW_2025_paper.pdf) — segment-informed tokens for efficient translation; up to 50% shorter inputs, 2.67x lower memory.
- [Subtitle–signing alignment](https://arxiv.org/abs/2512.08094) — segmentation as an alignment signal.
- [Extracting signs from weakly aligned corpora](https://www.sign-lang.uni-hamburg.de/lrec/pub/26039.html) — a study on LSF and LSM (sign-lang@LREC 2026).

Our interest is how well these trained models generalise:

- The original model was trained and evaluated on DGS; we have a small study
  transferring to LSF, both zero-shot and fine-tuned.
- Colleagues report it also works for ASL and NGT, and it has been useful for
  alignment in SEA for BSL.
- Beyond more languages, we want to cover edge cases: very short input clips
  (ISLR datasets), and false positives when no signer is present.
- Special linguistic phenomena such as fingerspelling and indexing may need
  dedicated treatment.

The aim is a systematic benchmark for the task, and a way to track future
progress from it.

## Related work

- Segment Anything (SAM 3)
- Segment Any Text

## Data

Datasets with gloss annotations, to be curated in [`datasets/`](datasets/):

- Public DGS Corpus
- BSL Corpus (TODO: request access)
- More to be researched, as part of an initial literature review phase

We can also annotate our own Swiss data (@agoehring): SwissSLi, Signsuisse.

## Method

Start by benchmarking the 2023 model
([sign-language-processing/segmentation](https://github.com/sign-language-processing/segmentation)),
where @AmitMY has already explored autoresearch to improve segmentation scores.
Then implement iterative, targeted improvements against the new benchmarks —
until we can confidently say our model can segment any sign.

## Layout

```
datasets/       one folder per corpus — curation and analysis
literature/     material for the related-work section
environment.yml conda env (`sas`)
```

## Setup

```bash
module load miniforge3
conda env create -f environment.yml
conda activate sas
```

Deliberately small: the dataset scripts read ELAN annotations directly and need
no TensorFlow, torch or pose-format. When we start running the model itself, add
`- -e ..` to the pip section of `environment.yml`.

## Data exploration

Analyses are written as percent-format Python (`# %%` cells), not notebooks, so
they diff and review cleanly in git. Open e.g.
[`datasets/public_dgs_corpus/explore.py`](datasets/public_dgs_corpus/explore.py)
in VS Code and run the cells against the `sas` interpreter, or use the
Interactive Window.

To regenerate a report, from this directory:

```bash
jupytext --to ipynb --execute datasets/public_dgs_corpus/explore.py -o - \
  | jupyter nbconvert --stdin --to markdown --output explore --output-dir datasets/public_dgs_corpus
```

This executes the file top-to-bottom in a fresh kernel, so the report always
matches the committed source, and no intermediate `.ipynb` is written. It
produces `explore.md` plus an `explore_files/` folder of plot images; both are
tracked, and **GitHub renders them inline** — which HTML would not be.

> **Note on the DGS data.** The 2023 TFDS build stores annotations as *paths*
> into `/shares/volk.cl.uzh/...`, which we no longer have permission to read.
> The same files survive in a backup under `~/sp2/zifjia/backups/`, which is what
> the scripts point at. See the header of `explore.py` for details.
