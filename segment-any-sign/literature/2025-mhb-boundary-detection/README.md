# MHB: Multimodal Handshape-aware Boundary Detection for Continuous Sign Language Recognition

- **Authors:** Mingyu Zhao, Zhanfu Yang, Yang Zhou, Zhaoyang Xia, Can Jin, Xiaoxiao He, Dimitris N. Metaxas
- **Venue:** arXiv, November 2025
- **Link:** https://arxiv.org/abs/2511.19907
- **Status:** not yet read

## Why it is here

The most direct external comparison point we have for ASL. It does **boundary
detection on continuous signing** — our task — and evaluates on the **ASLLRP
corpus**, which is where our NCSLGR data comes from. See
[`../../datasets/ncslgr/`](../../datasets/ncslgr/).

Two reasons it matters beyond being a baseline:

- It validates the corpus choice. When we surveyed ASL options, ASLLRP was the
  only resource with per-sign gloss boundaries; this is independent evidence
  that the ASL boundary-detection community is working on the same data.
- It comes from Metaxas's group at Rutgers — the same institution that hosts the
  ASLLRP Data Access Interface, so they are close to the annotation itself.

## Notes

- Trains sign recognition on a mix of citation-form isolated signs and signs
  pre-segmented from continuous signing using manual annotations.
- Uses a handshape classifier over 87 categories, built by integrating and
  normalising several existing datasets.
- _TODO: read for the exact metrics and splits — if they are comparable to ours,
  this is a direct number to beat or match on ASL._
