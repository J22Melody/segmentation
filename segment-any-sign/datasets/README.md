# Datasets

One folder per corpus, each with an `explore.py` and a generated `explore.md`.
Raw data stays on the shared filesystem, never in this repo.

A corpus used by the benchmark also gets a `load.py`: **one clip list, one set of
gold annotations, shared by every model.** Only
[`public_dgs_corpus/load.py`](public_dgs_corpus/load.py) exists so far. Models
differ in how they preprocess a pose before it enters the network — that part
lives in [`../benchmark/`](../benchmark/) — but never in which clips they see or
what they are scored against.

The benchmark needs **per-sign boundaries** — that is the dividing line below.

| dataset | language | sign-level | scale | state |
|---|---|---|---|---|
| [public_dgs_corpus](public_dgs_corpus/) | DGS | ✅ | 350,168 glosses | ready |
| [ncslgr](ncslgr/) | ASL | ✅ | 11,854 tokens | videos local; full XML needs a DAI account |
| [bsl_corpus](bsl_corpus/) | BSL | ✅ (subset) | 6,879 segments, 6.2 h | metadata only; awaiting video from UCL |
| [signsuisse](signsuisse/) | DSGS | 🔜 | 500 examples, 48 min | ELAN prepared; annotation not started |
| [how2sign](how2sign/) | ASL | ❌ phrase only | 79 h | glosses sentence-timed and unreleased |
| [mediapi_skel](mediapi_skel/) | LSF | ❌ none | 27 h | subtitle alignment only |

## Where the data lives

```
/shares/sign-language.ebling.cl.uzh/NCSLGR/
/shares/sign-language.ebling.cl.uzh/BSL_Corpus/
/shares/sign-language.ebling.cl.uzh/Signsuisse/
/shares/iict-sp2.ebling.cl.uzh/common/How2Sign/
/shares/iict-sp2.ebling.cl.uzh/common/tensorflow_datasets/{dgs_corpus,mediapi_skel}/
/shares/iict-sp2.ebling.cl.uzh/zifjia/backups/tensorflow_datasets_2/downloads/  (DGS eaf/cmdi)
```

## Open blockers

- DAI account (free) — unlocks NCSLGR's full XML and the ASLLRP SignStream 3 corpus
- UCL — BSL Corpus video, 239 files
- Gloss annotation of the 500 SignSuisse examples

Licensing differs per corpus and is recorded in each `explore.md`. Internal
research use is fine; a public release is not covered without permission.
