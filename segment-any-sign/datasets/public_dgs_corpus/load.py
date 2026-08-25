"""Shared Public DGS Corpus loading.

**One clip list, one set of gold annotations, every model.** Without this the
models drift apart on things that have nothing to do with modelling — a different
joke filter, a different split file, a different person-level flattening — and the
table stops comparing what it claims to compare.

The source is the TFDS build at `TFDS_ROOT`, read through the *original v2023*
`tfds_dataset.py`, because that is what produced the reproduction we verified. It
is vendored out of git into `.cache/v2023_src/` and patched only for imports; see
`../../benchmark/predict_dgs_2023.py` for why the original code is driven
rather than rewritten.

Two entry points, split by how much processing the caller wants:

  * `vendored()` returns the patched v2023 `data` module, for the 2023 model,
    which needs that pipeline end to end.
  * `iter_clips()` returns raw `Pose` objects plus gold spans **in seconds**,
    stopping before any model-specific preprocessing. That is the entry point for
    the 2026 model and anything after it.

Gold spans stay in seconds here on purpose. Each model converts them to frames
using its own protocol — v2023 floors both bounds, the 2026 code walks frame
timestamps — and those conversions differ by up to a frame. Forcing one of them on
the other model would break the reproduction it is meant to be checked against, so
the conversion is left to the caller and documented in `../../benchmark/README.md`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

TFDS_ROOT = "/shares/iict-sp2.ebling.cl.uzh/common/tensorflow_datasets"
# The TFDS records point at /shares/volk.cl.uzh/... which we can no longer read,
# so ELAN and CMDI files are re-resolved by basename into this backup.
BACKUP = "/shares/iict-sp2.ebling.cl.uzh/zifjia/backups/tensorflow_datasets_2/downloads"

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE = REPO_ROOT / ".cache"
SRC = CACHE / "v2023_src"

# The TFDS build on this cluster is `holistic-25`, so that route serves 25fps
# only. `iter_clips_native` reads the archived .pose downloads instead, which are
# the untouched 50fps originals TFDS was built from — no rebuild required.
AVAILABLE_FPS = 25
NATIVE_FPS = 50

COMPONENTS = ["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]

# Documents excluded by v2023 `data.py`, unchanged by the 2026 code.
EXCLUDED_IDS = ["1289910", "1245887", "1289868", "1246064", "1584617"]

# file -> git ref it comes from
VENDORED = {
    "data.py": "v2023",
    "pose_utils.py": "e3a020b",
    "tfds_dataset.py": "e3a020b",
    "utils/probs_to_segments.py": "v2023",
    "utils/__init__.py": "v2023",
}


def git_show(ref: str, path: str) -> bytes:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "show", f"{ref}:{path}"])
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"cannot read {path} from {ref}. Fetch the tag with\n"
            f"  git fetch https://github.com/sign-language-processing/segmentation "
            f"'+refs/tags/v2023:refs/tags/v2023'") from error


def vendor_v2023() -> None:
    """Materialise the original source under .cache/, patching only imports."""
    if (SRC / "data.py").exists():
        return
    (SRC / "utils").mkdir(parents=True, exist_ok=True)
    for name, ref in VENDORED.items():
        text = git_show(ref, f"sign_language_segmentation/src/{name}").decode()
        text = text.replace("from _shared.tfds_dataset import", "from tfds_dataset import")
        text = text.replace("from .pose_utils import", "from pose_utils import")
        (SRC / name).write_text(text)
    (SRC / "__init__.py").touch()
    print(f"vendored the v2023 pipeline into {SRC}")


def stub_mediapipe() -> None:
    """tfds_dataset imports mediapipe only for reduce_face, which we never use."""
    holistic = types.ModuleType("mediapipe.solutions.holistic")
    holistic.FACEMESH_CONTOURS = []
    solutions = types.ModuleType("mediapipe.solutions")
    solutions.holistic = holistic
    mediapipe = types.ModuleType("mediapipe")
    mediapipe.solutions = solutions
    sys.modules.setdefault("mediapipe", mediapipe)
    sys.modules.setdefault("mediapipe.solutions", solutions)
    sys.modules.setdefault("mediapipe.solutions.holistic", holistic)


def local(path: str, backup: str = BACKUP) -> str:
    """Re-resolve an absolute /shares/volk.cl.uzh path into our backup."""
    return os.path.join(backup, os.path.basename(path))


def vendored(backup: str = BACKUP):
    """Import the v2023 `data` module, with annotation lookups redirected.

    Returns the module itself so a caller can use the whole original pipeline.
    `get_elan_sentences` and `filter_dataset` are wrapped to read our backup;
    nothing else is touched.
    """
    vendor_v2023()
    stub_mediapipe()
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    import data as v2023  # the original pipeline

    if getattr(v2023, "_redirected", False):
        return v2023

    original_get_elan = v2023.get_elan_sentences
    v2023.get_elan_sentences = lambda path: original_get_elan(local(path, backup))

    def filter_dataset(tf_datum) -> bool:
        """v2023 filter_dataset, with the CMDI path re-resolved.

        Drops the five excluded documents and anything tagged as a joke — the
        latter is what takes the test split from 10 documents to 9.
        """
        if "paths" not in tf_datum:
            return True
        if tf_datum["id"].numpy().decode("utf-8") in EXCLUDED_IDS:
            return False
        with open(local(tf_datum["paths"]["cmdi"].numpy().decode("utf-8"), backup)) as f:
            return "<cmdp:Task>Joke</cmdp:Task>" not in f.read()

    v2023.filter_dataset = filter_dataset
    v2023._redirected = True
    return v2023


def iter_clips(split: str = "test", fps: int = AVAILABLE_FPS,
               tfds_root: str = TFDS_ROOT, backup: str = BACKUP):
    """Yield one dict per signer-video, with the pose unprocessed.

    Keys: `id` (`<document>_<person>`), `pose` (a `pose_format.Pose` holding only
    the body and hand components), `fps`, and `sentences`.

    Each sentence carries its **own** `start_time` / `end_time` — the German
    translation tier's timeslots — alongside its `glosses`, all in seconds. Both
    are kept because the two models disagree about what a phrase is, and the
    disagreement is not cosmetic: see `sign_phrase_spans`.

    Sentences with no glosses are dropped, matching both models' loaders.
    """
    v2023 = vendored(backup)

    import tfds_dataset

    data = tfds_dataset.get_tfds_dataset(
        name="dgs_corpus", poses="holistic", fps=fps, split=split,
        components=COMPONENTS, data_dir=tfds_root, filter_func=v2023.filter_dataset)

    for datum in data:
        poses = datum["pose"]
        elan_path = datum["tf_datum"]["paths"]["eaf"].numpy().decode("utf-8")
        sentences = list(v2023.get_elan_sentences(elan_path))

        for person in ("a", "b"):
            if len(poses[person].body.data) == 0:
                continue
            person_sentences = [
                {"start_time": s["start"] / 1000, "end_time": s["end"] / 1000,
                 "glosses": [{"start_time": g["start"] / 1000, "end_time": g["end"] / 1000}
                             for g in s["glosses"]]}
                for s in sentences
                if s["participant"].lower() == person and len(s["glosses"]) > 0
            ]
            yield {"id": f"{datum['id']}_{person}", "pose": poses[person],
                   "fps": fps, "sentences": person_sentences}


def sign_phrase_spans(sentences, phrase: str = "sentence") -> dict:
    """Derive sign and phrase gold spans from the shared sentence records.

    Sign spans are the glosses either way — the models agree there. Phrase spans
    do not agree, and the choice matters a great deal:

      * `phrase="sentence"` uses the sentence's own annotated bounds. This is what
        the 2026 loader does, and what its model was trained to predict.
      * `phrase="glosses"` runs from a sentence's first gloss to its last, which
        is how v2023 `build_classes_vectors` derives it.

    The gloss extent sits *inside* the annotated sentence, so the two differ by
    the lead-in and trail-out around the signing. Scoring the 2026 model against
    the gloss extent makes its near-contiguous sentence predictions look like they
    merge phrases, and costs it most of its phrase score.
    """
    if phrase not in ("sentence", "glosses"):
        raise ValueError(f"phrase must be 'sentence' or 'glosses', got {phrase!r}")

    signs = [gloss for sentence in sentences for gloss in sentence["glosses"]]
    if phrase == "sentence":
        phrases = [{"start_time": s["start_time"], "end_time": s["end_time"]}
                   for s in sentences]
    else:
        phrases = [{"start_time": s["glosses"][0]["start_time"],
                    "end_time": s["glosses"][-1]["end_time"]}
                   for s in sentences if s["glosses"]]
    return {"sign": signs, "phrase": phrases}


# -- native source: the archived .pose downloads ------------------------------
#
# The TFDS build downsampled poses to 25fps when it was made, and that is baked
# into the records. The 2026 model publishes 50fps numbers and reads raw .pose
# files directly, so serving it from TFDS would silently halve its input rate.
#
# The originals are still in the download archive, keyed by `original_fname` in
# each `.INFO` sidecar (`<document>_<person>.pose`, `<document>-*.eaf`, `.cmdi`).
# That is enough to rebuild the whole clip list without TFDS at all.


def _archive_index(backup: str = BACKUP) -> dict:
    """Map file kind -> key -> local path, read from the .INFO sidecars.

    Poses are keyed `<document>_<person>`; eaf and cmdi by document id, whose
    filenames carry extra session numbers after the id.
    """
    import glob
    import json

    index = {"pose": {}, "eaf": {}, "cmdi": {}}
    for info_path in glob.glob(os.path.join(backup, "*.INFO")):
        try:
            with open(info_path) as f:
                name = json.load(f).get("original_fname", "")
        except (OSError, ValueError):
            continue
        local_path = info_path[: -len(".INFO")]
        if not os.path.exists(local_path):
            continue
        stem, _, kind = name.rpartition(".")
        if kind == "pose":
            index["pose"][stem] = local_path
        elif kind in ("eaf", "cmdi"):
            index[kind][stem.split("-")[0]] = local_path
    return index


def _splits(splits_path: str | None = None) -> dict:
    """Read the 2026 split file, which extends split.3.0.0-uzh-document.

    Same dev/test document ids the 2023 TFDS split uses, so both routes select
    the same clips.
    """
    import json

    if splits_path is None:
        import sign_language_segmentation
        splits_path = (Path(sign_language_segmentation.__file__).parent /
                       "datasets" / "dgs" / "splits.json")
    return json.loads(Path(splits_path).read_text())


def iter_clips_native(split: str = "test", backup: str = BACKUP,
                      splits_path: str | None = None):
    """Yield clips from the archived .pose downloads, at their native 50fps.

    Same shape as `iter_clips`, and the same filters — the five excluded
    documents and anything tagged as a joke — so the clip list matches. The
    difference is the pose: untouched originals rather than the TFDS build's
    25fps downsample.
    """
    from pose_format import Pose

    from sign_language_segmentation.datasets.dgs.utils import get_elan_sentences

    index = _archive_index(backup)
    splits = _splits(splits_path)
    wanted = set(splits["dev" if split in ("dev", "validation") else split]) \
        if split in ("dev", "validation", "test") else None

    for doc_id in sorted(index["eaf"]):
        if doc_id in EXCLUDED_IDS:
            continue
        if wanted is not None and doc_id not in wanted:
            continue
        cmdi_path = index["cmdi"].get(doc_id)
        if cmdi_path is None:
            continue
        with open(cmdi_path) as f:
            if "<cmdp:Task>Joke</cmdp:Task>" in f.read():
                continue

        sentences = list(get_elan_sentences(index["eaf"][doc_id]))

        for person in ("a", "b"):
            pose_path = index["pose"].get(f"{doc_id}_{person}")
            if pose_path is None:
                continue
            person_sentences = [
                {"start_time": s["start"] / 1000, "end_time": s["end"] / 1000,
                 "glosses": [{"start_time": g["start"] / 1000, "end_time": g["end"] / 1000}
                             for g in s["glosses"]]}
                for s in sentences
                if s["participant"].lower() == person and len(s["glosses"]) > 0
            ]
            with open(pose_path, "rb") as f:
                pose = Pose.read(f)
            yield {"id": f"{doc_id}_{person}", "pose": pose,
                   "fps": float(pose.body.fps), "sentences": person_sentences}
