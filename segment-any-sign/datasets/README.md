# Datasets — cross-dataset overview

One folder per corpus. Each contains an `explore.py` (percent-format, run the
`# %%` cells in VS Code) and a generated `explore.md` with the numbers and plots.
Raw data never lives in this repo — it stays on the shared filesystem.

## Can it support sign-level segmentation?

The benchmark needs **per-sign boundaries**. That is the dividing line below, and
it rules out more corpora than expected — large ASL datasets in particular tend
to be aligned at subtitle or sentence level only.

| dataset | language | sign-level | scale | state |
|---|---|---|---|---|
| [public_dgs_corpus](public_dgs_corpus/) | DGS | ✅ | 313 docs · 63,672 sentences · **350,168 glosses** | ready; reproduces the 2023 paper's splits |
| [ncslgr](ncslgr/) | ASL | ✅ | 1,887 utterances · 11,854 tokens | 2,636 videos (4.7 GB) local; **full XML needs a DAI account** |
| [bsl_corpus](bsl_corpus/) | BSL | ✅ (subset) | 6,879 segments · 6.2 h · 198 signers | metadata only; **awaiting video from UCL** |
| [signsuisse](signsuisse/) | DSGS | 🔜 to annotate | 500 examples · 48.1 min | ELAN files prepared; gloss annotation not started |
| [how2sign](how2sign/) | ASL | ❌ phrase only | 79.1 h · 35,191 sentences | complete copy local; glosses sentence-timed **and unreleased** |
| [mediapi_skel](mediapi_skel/) | LSF | ❌ none | 27 h · 368 videos | TFDS build local; subtitle alignment only |

Planned but not started: **ASLLRP SignStream 3 Corpus** — a separate corpus from
NCSLGR with richer annotation (both hands, sign type, handshapes), 2,127
utterances / 17,522 tokens. Needs the same DAI account. It should get its own
`asllrp_signstream3/` rather than being folded into `ncslgr/`.

## Where the data lives

| dataset | path |
|---|---|
| public_dgs_corpus | `/shares/iict-sp2.ebling.cl.uzh/common/tensorflow_datasets/dgs_corpus/` (poses)<br>`/shares/iict-sp2.ebling.cl.uzh/zifjia/backups/tensorflow_datasets_2/downloads/` (eaf/cmdi) |
| ncslgr | `/shares/sign-language.ebling.cl.uzh/NCSLGR/` |
| bsl_corpus | `/shares/sign-language.ebling.cl.uzh/BSL_Corpus/` |
| signsuisse | `/shares/sign-language.ebling.cl.uzh/Signsuisse/` |
| how2sign | `/shares/iict-sp2.ebling.cl.uzh/common/How2Sign/` |
| mediapi_skel | `/shares/iict-sp2.ebling.cl.uzh/common/tensorflow_datasets/mediapi_skel/` |

Scripts use true `/shares/...` paths, never the `~/easier` or `~/sp2` symlinks,
so they resolve identically for anyone on the cluster.

## Conventions

- **Directories are named after the corpus, not the project** — hence `ncslgr`,
  not `asllrp` (ASLLRP distributes two distinct corpora).
- **Closed-access data is group-only.** `BSL_Corpus/` and `NCSLGR/` are
  `drwxrws---`, group `s3it_t_hpc_sign-language.ebling.cl.uzh`. Keep anything
  added that way.
- **Reports are markdown, not HTML** — GitHub renders `.md` with images inline.
  Regenerate from `segment-any-sign/`:
  ```bash
  jupytext --to ipynb --execute datasets/<name>/explore.py -o - \
    | jupyter nbconvert --stdin --to markdown --output explore --output-dir datasets/<name>
  ```

## Licensing — check before publishing anything derived

| dataset | terms |
|---|---|
| ncslgr | No named licence. Research/education only; **no redistribution without permission**; no commercial use; citation required ([terms](https://www.bu.edu/asllrp/dai-terms.html)) |
| bsl_corpus | Primo record marks rights `closed` alongside a CC BY-NC-SA 2.0 UK string; the derived index we hold is Renz et al.'s, whose terms are separate again |
| signsuisse | WMT-SLT / Signsuisse terms; shared space owned by another user |
| public_dgs_corpus | Public DGS Corpus terms |

Internal research use is fine throughout. A **public benchmark release is not**
covered by any of these without explicit permission, and for NCSLGR and BSL
Corpus that permission has to be requested from the data owners. Worth settling
before building a release on top of them, not after.

## Open blockers

- [ ] **DAI account** (free, person-tied) — unlocks NCSLGR's full XML and the
      SignStream 3 corpus.
- [ ] **UCL** — BSL Corpus source video; 239 files, manifest already known.
- [ ] **Oxford/VGG** — firewall blocks SSH, so the earlier BSL derivatives copy
      is unreachable; may make the UCL request unnecessary if recovered.
- [ ] **Gloss annotation** of the 500 SignSuisse DSGS examples.

## Gotchas worth knowing before using any of these

These each cost real time to discover, and each would quietly corrupt results:

- **NCSLGR timescale is 2000 units/second, not milliseconds.** Nothing documents
  it; it was established by measuring annotations against video duration.
  Halving it makes gloss durations (~200 ms) line up with DGS.
- **How2Sign's `eaf_files` are model output, not gold** — authored by
  `sign-langauge-processing/transcription` with empty annotation values. Using
  them as ground truth would evaluate the 2023 model against itself.
- **BSL Corpus annotation has almost no background class** — only 0.7% of frames
  are `SILENCE`; signs tile the timeline. A model that legitimately predicts `O`
  will look wrong there.
- **MEDIAPI-SKEL's sign tier is synthetic** in the old codebase: one fake "sign"
  per subtitle, so any historical sign-F1 on it measures subtitle units.
- **DGS drops 88 documents as "Joke"** plus 5 hardcoded IDs — 313 of 406. Any
  published DGS number must state this.
