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

# The only DGS config built on this cluster is `holistic-25`, so 25 is the only
# fps we can actually serve. The 2026 model's published numbers are at 50fps;
# see ../../benchmark/README.md for what that costs and what a 50fps build takes.
AVAILABLE_FPS = 25

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
    the body and hand components), `fps`, and `sign` / `phrase` gold spans as
    `{"start_time", "end_time"}` in **seconds**.

    Sign spans are the glosses; a phrase span runs from the first gloss of a
    sentence to its last, which is how v2023 `build_classes_vectors` derives them.
    Sentences with no glosses are dropped upstream, so both levels come from the
    same annotations.
    """
    v2023 = vendored(backup)

    import tfds_dataset

    data = tfds_dataset.get_tfds_dataset(
        name="dgs_corpus", poses="holistic", fps=fps, split=split,
        components=COMPONENTS, data_dir=tfds_root, filter_func=v2023.filter_dataset)

    for datum in data:
        for item in v2023.process_datum_dgs_corpus(datum):
            # item["segments"] is a list of sentences, each a list of gloss spans
            signs = [gloss for sentence in item["segments"] for gloss in sentence]
            phrases = [{"start_time": sentence[0]["start_time"],
                        "end_time": sentence[-1]["end_time"]}
                       for sentence in item["segments"] if sentence]
            yield {"id": item["id"], "pose": item["pose"], "fps": fps,
                   "sign": signs, "phrase": phrases}
