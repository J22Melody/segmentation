# MHB: Multimodal Handshape-aware Boundary Detection for Continuous Sign Language Recognition

- **Authors:** Mingyu Zhao, Zhanfu Yang, Yang Zhou, Zhaoyang Xia, Can Jin, Xiaoxiao He, Dimitris N. Metaxas
- **Venue:** arXiv, November 2025
- **Link:** https://arxiv.org/abs/2511.19907
- **Status:** not yet read

## Why it is here

Follow-up work on boundary detection in continuous signing, for ASL. It
evaluates on the ASLLRP corpus, which is the source of our NCSLGR data — see
[`../../datasets/ncslgr/`](../../datasets/ncslgr/).

## Notes

From the abstract (not yet verified against the full paper):

- Evaluated on the ASLLRP corpus.
- The sign recognition model is trained on both citation-form isolated signs and
  signs pre-segmented from continuous signing using manual annotations.
- Uses a handshape classifier over 87 categories, built by integrating and
  normalising several existing datasets.

_TODO: read for the metrics, splits, and whether the numbers are comparable to
ours._
