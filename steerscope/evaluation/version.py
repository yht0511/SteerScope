"""Version identifiers for persisted evaluation results and inference caches."""

import hashlib
import inspect
import os
from functools import lru_cache
from pathlib import Path


EVALUATION_PIPELINE_VERSION = "4"
PERPLEXITY_SCORING_VERSION = "base-continuation-v1"


_SOURCE_ROOTS = (
    "evaluation",
    "inference",
    "models",
    "templates",
)
_SOURCE_FILES = (
    "__init__.py",
    "scripts/evaluate.py",
    "scripts/args/eval_args.py",
    "scripts/args/training_args.py",
    "utils/constants.py",
    "utils/data_utils.py",
    "utils/model_utils.py",
)


def _source_fingerprint(package_dir):
    package_dir = Path(package_dir)
    files = [package_dir / relative for relative in _SOURCE_FILES]
    for relative in _SOURCE_ROOTS:
        files.extend((package_dir / relative).rglob("*.py"))

    digest = hashlib.sha256()
    for path in sorted(set(files)):
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(package_dir)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def evaluation_source_fingerprint():
    """Hash source files that can change evaluation or model inference behavior."""
    compatible = os.environ.get("STEERSCOPE_EVALUATION_COMPAT_FINGERPRINT")
    if compatible is not None:
        compatible = compatible.strip().lower()
        if len(compatible) != 64 or any(
            character not in "0123456789abcdef" for character in compatible
        ):
            raise ValueError(
                "STEERSCOPE_EVALUATION_COMPAT_FINGERPRINT must be a SHA-256 hex digest."
            )
        return compatible
    package_dir = Path(__file__).resolve().parents[1]
    return _source_fingerprint(package_dir)


@lru_cache(maxsize=None)
def evaluator_source_fingerprint(evaluator_class):
    """Hash one evaluator's implementation, bases, and explicit dependencies."""
    package_dir = Path(__file__).resolve().parents[1]
    files = set()
    for base in evaluator_class.__mro__:
        if not base.__module__.startswith("steerscope.evaluators"):
            continue
        source = inspect.getsourcefile(base)
        if source is not None:
            files.add(Path(source).resolve())
        for relative in base.__dict__.get("source_dependencies", ()):
            files.add((package_dir / relative).resolve())
    return _file_set_fingerprint(files, package_dir)


def _file_set_fingerprint(files, package_dir):
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        if not path.is_file():
            continue
        try:
            label = path.relative_to(package_dir)
        except ValueError:
            label = path
        digest.update(str(label).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def file_signature(path):
    """Return a content-based signature for an optional evaluator input file."""
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def path_signature(path):
    """Return content signatures for a file or directory tree."""
    path = Path(path)
    if not path.exists():
        return None
    if path.is_file():
        return file_signature(path)
    files = {}
    for root, directories, names in os.walk(path, followlinks=True):
        directories.sort()
        for name in sorted(names):
            child = Path(root) / name
            files[str(child.relative_to(path))] = file_signature(child)
    return {"path": str(path), "files": files}


def generated_data_identity(
    data_dir,
    *,
    metadata_dir=None,
    use_dpo_loss=False,
):
    """Hash generated-pool content with root-relative paths so moved pools retain their identity."""
    data_dir = Path(data_dir)
    metadata_dir = Path(metadata_dir) if metadata_dir is not None else data_dir
    metadata_path = metadata_dir / "metadata.jsonl"
    data_prefix = "dpo_train_data" if use_dpo_loss else "train_data"
    data_paths = sorted(
        path
        for path in data_dir.glob(f"{data_prefix}*.parquet")
        if "combined" not in path.name
    )
    missing = []
    if not metadata_path.is_file():
        missing.append(str(metadata_path))
    if not data_paths:
        missing.append(str(data_dir / f"{data_prefix}*.parquet"))
    if missing:
        raise FileNotFoundError(
            "Generated data identity is missing required inputs: "
            f"{missing}."
        )

    def content(path):
        signature = file_signature(path)
        return {
            "name": path.name,
            "size": signature["size"],
            "sha256": signature["sha256"],
        }

    return {
        "version": 1,
        "data_prefix": data_prefix,
        "metadata": content(metadata_path),
        "data_files": [content(path) for path in data_paths],
    }
