"""Score prediction files and print the benchmark table.

Stage 2 of 2. Reads the JSON written by a `predict_*.py` script and applies
[`../metrics/`](../metrics/). Kept separate from inference so that changing a
metric only means re-scoring, never re-running a model.

Environment: **`sas`** — this needs only numpy and scikit-learn, no torch and no
TensorFlow.

    conda activate sas
    python benchmark/score.py benchmark/predictions/*.json
    python benchmark/score.py benchmark/predictions/*.json --per-clip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from metrics import (aggregate, evaluate_level, frame_f1,  # noqa: E402
                     frame_f1_micro, global_iou, segment_counts,
                     segment_percentage)  # evaluate_level is used by --per-clip

LEVELS = ("sign", "phrase")
COLUMNS = ("frame_f1", "frame_f1_micro", "iou", "percentage", "mf1s")


def unrle(runs) -> np.ndarray:
    """Expand run-length encoded frame labels."""
    return np.concatenate([np.full(count, value, dtype=np.int64) for value, count in runs])


def score_clip(clip: dict, level: str) -> dict:
    """Metrics for one clip at one level, matching how the 2023 eval computed them.

    Frame metrics come from the stored per-frame labels — gold from `build_bio`,
    prediction from the argmax of the probabilities — because that is what the
    2023 code scored. Segment metrics come from the decoded segments against
    `floor`-converted gold. The two use different golds; that asymmetry is the
    original's, reproduced rather than tidied away.

    Prediction files must carry frame labels. Deriving them from the decoded
    segments instead would silently compute a different quantity, so that is an
    error rather than a fallback.
    """
    pred, gold = clip["pred"].get(level, []), clip["gold"].get(level, [])
    num_frames = clip["num_frames"]

    if "gold_bio" not in clip or "pred_bio" not in clip:
        raise SystemExit(
            f"clip {clip.get('id')} has no frame labels; re-run the predict step")

    gold_bio, pred_bio = unrle(clip["gold_bio"][level]), unrle(clip["pred_bio"][level])
    # labels=None reproduces the 2023 evaluation, which called sklearn without a
    # label set and so averaged only over classes present. A clip with no
    # annotation, predicted empty, scores 1.0 rather than 0.33 — worth 0.12 on
    # the DGS test mean.
    result = {"frame_f1": frame_f1(pred_bio, gold_bio, labels=None),
              "frame_f1_micro": frame_f1_micro(pred_bio, gold_bio, labels=None)}

    result["iou"] = global_iou(pred, gold, num_frames)
    result["percentage"] = segment_percentage(pred, gold)
    result["mf1s_counts"] = segment_counts(pred, gold)
    result["mf1s"] = 0.0  # per-clip value unused; mF1S is aggregated micro
    return result


def score_file(path: Path) -> dict:
    """Score one prediction file, per level."""
    payload = json.loads(path.read_text())
    clips = payload["clips"]

    results = {}
    for level in LEVELS:
        # A level is scored when the dataset annotates it at all. Individual
        # clips with no annotation are still scored — the 2023 evaluation kept
        # them, and dropping them shifts the corpus mean. A level no clip
        # annotates stays out of the table entirely, blank rather than zero.
        if not any(clip["gold"].get(level) for clip in clips):
            results[level] = None
            continue
        results[level] = aggregate([score_clip(clip, level) for clip in clips])

    return {
        "model": payload.get("model", path.stem),
        "dataset": payload.get("dataset", "?"),
        "split": payload.get("split", "?"),
        "thresholds": payload.get("thresholds", {}),
        "clips": len(clips),
        "levels": results,
    }


def format_table(rows: list[dict]) -> str:
    """One row per model, with Sign and Phrase as column groups.

    Laid out like Table 2 of the 2023 paper, which groups the levels side by
    side rather than stacking them, so two models are compared at a glance.
    Decoding thresholds are shown per level, since they differ (the tuned phrase
    setting is 90/90 for E1s but 80/80 for E4s) and the scores move with them.
    """
    metrics = ("frame_f1", "frame_f1_micro", "iou", "percentage", "mf1s")
    short = {"frame_f1": "F1", "frame_f1_micro": "acc", "iou": "IoU",
             "percentage": "%", "mf1s": "mF1S"}

    group = f"{'b/o':>7} " + " ".join(f"{short[m]:>6}" for m in metrics)
    width = len(group)
    head1 = f"{'model':<14} {'dataset':<18} {'split':<5} {'clips':>5} │ " \
            f"{'Sign'.center(width)} │ {'Phrase'.center(width)}"
    head2 = f"{'':<14} {'':<18} {'':<5} {'':>5} │ {group} │ {group}"
    rule = "─" * len(head1)

    lines = [rule, head1, head2, rule]
    for row in rows:
        cells = []
        for level in LEVELS:
            values = row["levels"].get(level)
            pair = row["thresholds"].get(level)
            if values is None:
                # the dataset does not annotate this level: blank, not zero
                cells.append(f"{'—':>7} " + " ".join(f"{'—':>6}" for _ in metrics))
                continue
            label = f"{pair[0]:g}/{pair[1]:g}" if pair else "-"
            cells.append(f"{label:>7} " + " ".join(f"{values[m]:>6.3f}" for m in metrics))
        lines.append(
            f"{row['model']:<14} {row['dataset']:<18} {row['split']:<5} "
            f"{row['clips']:>5} │ {cells[0]} │ {cells[1]}")
    lines.append(rule)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--per-clip", action="store_true",
                        help="also print each clip, worst mF1S first")
    parser.add_argument("--out", type=Path, default=None, help="write results as JSON")
    args = parser.parse_args()

    rows = [score_file(path) for path in args.predictions]
    print(format_table(rows))

    if args.per_clip:
        for path in args.predictions:
            payload = json.loads(path.read_text())
            print(f"\nper clip — {payload.get('model', path.stem)}")
            scored = []
            for clip in payload["clips"]:
                if not clip["gold"].get("sign"):
                    continue
                result = evaluate_level(clip["pred"].get("sign", []),
                                        clip["gold"]["sign"], clip["num_frames"])
                scored.append((result["mf1s"], clip["id"], result,
                               len(clip["gold"]["sign"]), len(clip["pred"].get("sign", []))))
            for mf1s_value, clip_id, result, n_gold, n_pred in sorted(scored):
                print(f"  {clip_id:<22} gold {n_gold:>5} pred {n_pred:>5}  "
                      f"F1 {result['frame_f1']:.4f}  IoU {result['iou']:.4f}  "
                      f"%={result['percentage']:.2f}  mF1S {mf1s_value:.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
