"""Launch a 2026-style training run on the benchmark's own DGS clips.

Thin wrapper. The training loop, model, augmentation and checkpointing are
upstream's — this only swaps the dataset for `dgs_shared`
([`dgs_dataset.py`](dgs_dataset.py)) so a run trains on exactly the clips,
filters and gold that [`../benchmark/`](../benchmark/) scores, and then calls
`sign_language_segmentation.train.train()` unchanged.

Every hyperparameter is upstream's `args.py`; anything not passed keeps its 2026
default, except for three we override: `--batch_size` 32 (upstream 8), `--epochs`
100 (200) and `--patience` 25 (10). These flags are ours and are stripped before
upstream parses:

    --phrase {glosses,sentence}   what counts as a phrase (default glosses, the
                                  benchmark's definition — see ../benchmark/)
    --dry-run                     build the data, model and one forward pass,
                                  then stop. No optimiser step, no wandb.
    --skip-stats                  skip the data report (it costs ~1 min)
    --no-test                     skip the end-of-training test evaluation
    --limit N                     use only the first N annotated clips per split.
                                  Turns a dry run into seconds. Never for a
                                  reported number.
    --select-on {mean_mf1s,hm_iou}
                                  what the best checkpoint maximises. Default is
                                  the mean of sign and phrase mF1S; upstream used
                                  hm_iou.

Every run first writes a data report — split sizes, segment counts, label balance
and correctness checks — to `dist/<run>/data_stats.{json,log}`, and attaches its
scalar summary to the W&B run. A run that leaks documents between splits aborts
before training rather than producing a number nobody can trust.

The full metric set from `../metrics/` is logged on **both** train and validation
under matching names, so W&B can overlay the two curves — see
[`validation_metrics.py`](validation_metrics.py).

Validation runs **every epoch** (upstream's default), and the best checkpoint
maximises `validation_mean_mf1s` — see `--select-on`. **Test runs once**, after training, on
that checkpoint — through `benchmark/`, so the number lands on the same protocol
as every row of the benchmark table. Results go to `dist/<run>/test_results.log`.

    conda activate sas
    python experiments/train.py --dry-run --device cpu --no_wandb

Note upstream's `--max_time` defaults to **30 minutes**, which will silently cut
a real run short; pass it explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    # Take our flags out of argv first: upstream's args.py calls parse_args() at
    # import time and rejects anything it does not know.
    ours = argparse.ArgumentParser(add_help=False)
    ours.add_argument("--phrase", default="glosses", choices=["glosses", "sentence"])
    ours.add_argument("--dry-run", action="store_true")
    ours.add_argument("--skip-stats", action="store_true")
    ours.add_argument("--no-test", action="store_true")
    ours.add_argument("--limit", type=int, default=None)
    ours.add_argument("--select-on", default="mean_mf1s",
                      choices=["mean_mf1s", "hm_iou"],
                      help="checkpoint selection metric (default mean_mf1s: the "
                           "mean of sign and phrase mF1S; hm_iou is upstream's)")
    mine, rest = ours.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    # registers "dgs_shared" before anything asks the registry for it
    from experiments import dgs_dataset  # noqa: F401

    from datasets.public_dgs_corpus import load as dgs_data
    from sign_language_segmentation.args import args

    # Upstream's args.py prints the parsed Namespace at import — before any of
    # this — so that first line still shows `datasets='all'` and the two
    # /mnt/nas GCS paths it defaults to. Neither is what we read. Correct them
    # here so the value logged to W&B is truthful, then restate the effective
    # configuration below, since the misleading line cannot be unprinted.
    args.phrase = mine.phrase
    args.limit = mine.limit
    if args.datasets == "all":
        # "all" would resolve to whatever happens to be registered; be explicit
        args.datasets = dgs_dataset.SharedDGSDataset.dataset_name
    args.corpus = args.poses = dgs_data.BACKUP
    args.data_loader = "datasets/public_dgs_corpus/load.py (archive, native fps)"

    print("\neffective data configuration"
          f"\n  dataset      {args.datasets}"
          f"\n  loader       {args.data_loader}"
          f"\n  archive      {dgs_data.BACKUP}"
          f"\n  phrase gold  {args.phrase}"
          + (f"\n  limit        {args.limit} clips per split (NOT a reportable run)"
             if args.limit else ""))

    import sign_language_segmentation.train as upstream_train
    from sign_language_segmentation.train import _dated_run_name, train

    # Upstream's train() instantiates whatever `PoseTaggingModel` names in its own
    # module namespace, so swapping the symbol there is enough to get the extra
    # metrics without touching upstream code or copying its training loop.
    from experiments.validation_metrics import ValidationMetricsModel
    upstream_train.PoseTaggingModel = ValidationMetricsModel

    # Batch 32 rather than upstream's 8: measured activations are 0.33 GiB per
    # sample at 1024 frames, so 32 is comfortable on a 40GB A100 and leaves room
    # on an 80GB one. Note this changes steps/epoch, and with it the OneCycle
    # schedule — a run at another batch size is not directly comparable.
    if "--batch_size" not in rest:
        args.batch_size = 32
    if "--epochs" not in rest:
        args.epochs = 100
    if "--patience" not in rest:
        args.patience = 25

    run_dir = Path("dist") / _dated_run_name(args.run_name)

    # Upstream names the directory `<run_name>-<YYYY.MM.DD>`, so a second run with
    # the same id on the same day lands in it — and Lightning writes best-v1.ckpt
    # beside the first run's best.ckpt rather than overwriting, silently mixing
    # two models in one directory. Refuse instead: one id, one run, one directory.
    if not mine.dry_run and list(run_dir.glob("*.ckpt")):
        raise SystemExit(
            f"{run_dir} already holds checkpoints. Give --run_name a new "
            f"experiment id (<NN>_<slug>) rather than reusing this one.")
    # the data report describes the full corpus, so it is meaningless under --limit
    if not mine.skip_stats and mine.limit is None:
        report = write_data_report(run_dir, phrase=mine.phrase)
        # flat scalars ride along into W&B via upstream's log_hyperparams
        for split, s in report["splits"].items():
            for key in ("documents", "videos", "hours", "signs", "phrases",
                        "unannotated_videos"):
                setattr(args, f"data_{split}_{key}", s[key])

    if mine.dry_run:
        dry_run(args)
        return

    monitor = {"mean_mf1s": "validation_mean_mf1s",
               "hm_iou": "validation_hm_iou"}[mine.select_on]
    print(f"  selection    {monitor} (max)\n")

    # train() takes monitor_metric, so both ModelCheckpoint and EarlyStopping
    # follow it without patching anything
    train(monitor_metric=monitor)

    if not mine.no_test:
        test_best_checkpoint(run_dir, phrase=mine.phrase)


def write_data_report(run_dir: Path, phrase: str) -> dict:
    """Write the data report beside the checkpoints, and abort on leakage."""
    from experiments import data_stats

    report = data_stats.collect(phrase=phrase)
    text = data_stats.format_report(report)
    print(text)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "data_stats.json").write_text(json.dumps(report, indent=2))
    (run_dir / "data_stats.log").write_text(text + "\n")
    print(f"\nwrote {run_dir}/data_stats.json and .log\n")

    if leakage := report["problems"]["leakage"]:
        raise SystemExit(f"ABORT — documents shared between splits: {leakage}")
    return report


def test_best_checkpoint(run_dir: Path, phrase: str) -> None:
    """Evaluate the best checkpoint on test, once, through the benchmark.

    Deliberately shells out to `benchmark/predict_dgs_2026.py` and `score.py`
    rather than scoring inline: the test number a run reports and the number in
    the benchmark table must come from the same code, or they will drift.
    """
    import subprocess

    # Lightning appends -v1, -v2 rather than overwriting, so a second run with
    # the same name and date leaves the *earlier* run's best.ckpt in place.
    # Take the newest and say which, or a rerun silently evaluates the old model.
    candidates = sorted(run_dir.glob("best*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        print(f"no best*.ckpt in {run_dir} — skipping test")
        return
    checkpoint = candidates[-1]
    if len(candidates) > 1:
        print(f"note: {len(candidates)} checkpoints here, testing the newest "
              f"({checkpoint.name}); others: {[c.name for c in candidates[:-1]]}")

    here = Path(__file__).resolve().parent
    predictions = here.parent / "benchmark" / "predictions" / f"{run_dir.name}.json"
    steps = [
        [sys.executable, str(here.parent / "benchmark" / "predict_dgs_2026.py"),
         "--split", "test", "--model", str(checkpoint), "--phrase", phrase,
         "--label", run_dir.name, "--out", str(predictions)],
        [sys.executable, str(here.parent / "benchmark" / "score.py"), str(predictions)],
    ]

    output = []
    for step in steps:
        print(f"\n$ {' '.join(step)}")
        result = subprocess.run(step, capture_output=True, text=True)
        print(result.stdout or result.stderr)
        output.append(result.stdout or result.stderr)
        if result.returncode != 0:
            print(f"test step failed ({result.returncode})")
            break

    (run_dir / "test_results.log").write_text("\n".join(output))
    print(f"wrote {run_dir}/test_results.log")


def dry_run(args) -> None:
    """Build data and model, run one forward pass, report shapes. No training.

    Checks the parts a wrapper can actually break — the item list, windowing,
    collation, and that the pose dimensions the dataset produces match what the
    model expects — without burning a GPU allocation to find out.
    """
    import torch

    from sign_language_segmentation.datasets.common import Split, get_dataloader
    from sign_language_segmentation.model.model import PoseTaggingModel

    loaders = {}
    for split, batch_size in ((Split.TRAIN, args.batch_size), (Split.DEV, 1)):
        loaders[split] = get_dataloader(split=split, dataset_names=args.datasets,
                                        args=args, batch_size=batch_size,
                                        persistent_workers=False)
        print(f"{split}: {len(loaders[split].dataset)} clips, "
              f"{len(loaders[split])} batches of {batch_size}")

    datum = loaders[Split.TRAIN].dataset[0]
    joints, dims = datum["pose"].shape[1:3]
    print(f"\nfirst clip     pose {tuple(datum['pose'].shape)}  "
          f"(joints={joints}, dims={dims})")
    for level in ("sign", "sentence"):
        bio = datum["bio"][level]
        counts = {int(v): int((bio == v).sum()) for v in bio.unique()}
        print(f"  {level:<9} BIO counts {counts}  (UNK=0, O=1, B=2, I=3)")

    batch = next(iter(loaders[Split.TRAIN]))
    print(f"\nbatch          pose {tuple(batch['pose'].shape)}  "
          f"lengths {batch['lengths'].tolist()}")

    model = PoseTaggingModel(
        pose_dims=(joints, dims), hidden_dim=args.hidden_dim,
        encoder_depth=args.encoder_depth, learning_rate=args.learning_rate,
        steps_per_epoch=len(loaders[Split.TRAIN]), max_epochs=args.epochs,
        dice_loss_weight=args.dice_loss_weight, optimizer=args.optimizer,
        attn_nhead=args.attn_nhead, attn_ff_mult=args.attn_ff_mult,
        attn_dropout=args.attn_dropout, fps_aug=args.fps_aug,
        frame_dropout=args.frame_dropout, num_frames=args.num_frames)
    print(f"parameters     {sum(p.numel() for p in model.parameters()):,}")

    model.eval()
    with torch.no_grad():
        out = model(batch["pose"], timestamps=batch.get("timestamps"))
    print(f"forward        sign {tuple(out['sign'].shape)}  "
          f"sentence {tuple(out['sentence'].shape)}")

    loss = model.step(batch, name="dry_run")
    print(f"loss           {float(loss):.4f}")
    print("\ndry run OK — data, model and loss all wire up")


if __name__ == "__main__":
    main()
