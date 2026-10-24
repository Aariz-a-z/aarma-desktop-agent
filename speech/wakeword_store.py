"""On-disk storage for wake-word models the user enrolled themselves.

A wake-word model is trained for one phrase, so "can AARMA hear this name?" is
answered by asking whether a model for that phrase exists.  Shipped models live
in ``models/`` beside the code; enrolled ones live here, under the user's data
directory:

    <AARMA_HOME>/models/custom/hey_friday/
        verifier.pt        stage-2 classifier, a torch state_dict
        reference.npz      stage-1 enrollment embeddings
        metadata.json      phrase, thresholds, validation, provenance

Enrolled models are **user data, not application files**.  They are recordings
of a particular person's voice in a particular room, they cannot be
regenerated from the repository, and on a real installation the application
directory may not even be writable.  So they sit next to the wake-word corpus
and the database, and a test that redirects ``AARMA_HOME`` gets isolation for
free.

Two safety rules shape this module.

*Paths are derived, never stored.*  ``metadata.json`` records what the model
*is*, and the directory it lives in is computed from the phrase.  A metadata
file cannot therefore point the loader at another location, however it was
edited or wherever it came from.

*Checkpoints are data, never code.*  They are read with ``weights_only=True``
and their tensor shapes are checked against the architecture before anything
is loaded, so a corrupt or hostile file fails as an unavailable detector rather
than as arbitrary code.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from config.settings import normalise_wake_phrase
from utils.logger import get_logger

logger = get_logger("speech.wakeword_store")

#: Bumped when the on-disk layout changes in a way older code cannot read.
SCHEMA_VERSION = 1

VERIFIER_NAME = "verifier.pt"
REFERENCE_NAME = "reference.npz"
METADATA_NAME = "metadata.json"

#: What the enrolled models are: the same two-stage cascade the shipped Aariz
#: detector uses, with its own enrollment reference and its own trained
#: verifier.  Recorded so a future format can be told apart from this one.
DETECTOR_TYPE = "cascade-efficientwordnet-farfield"

#: Slugs are the directory name, so they are restricted to characters that mean
#: the same thing on every filesystem and cannot climb out of the store.
_SLUG_OK = re.compile(r"^[a-z0-9](?:[a-z0-9_]{0,62}[a-z0-9])?$")

#: The verifier's shape.  A checkpoint that does not match is refused before
#: any of it reaches the model.
EXPECTED_MELS = 40
MIN_FRAMES, MAX_FRAMES = 16, 4096
MAX_PARAMETERS = 5_000_000
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024


class CustomWakeWordError(RuntimeError):
    """Raised when an enrolled model cannot be stored or read back."""


def slug_for_phrase(phrase: str) -> str:
    """Directory name for a wake phrase: "Hey Friday" -> ``hey_friday``.

    Derived from the normalised phrase so that the same phrase always lands in
    the same place, however the user capitalised or spaced it.
    """
    normalised = normalise_wake_phrase(phrase)
    slug = re.sub(r"[^a-z0-9]+", "_", normalised).strip("_")
    return slug


def is_safe_slug(slug: str) -> bool:
    return bool(slug) and bool(_SLUG_OK.match(slug))


@dataclass(frozen=True)
class CustomWakeWord:
    """One enrolled detector: where it is, and what it claims about itself."""

    slug: str
    directory: Path
    phrase: str
    display_name: str
    stage1_threshold: float
    stage2_threshold: float
    model_version: int = 1
    created_at: str = ""
    enrollment_count: int = 0
    detector_type: str = DETECTOR_TYPE
    validation: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    architecture: Dict[str, Any] = field(default_factory=dict)

    @property
    def verifier_path(self) -> Path:
        return self.directory / VERIFIER_NAME

    @property
    def reference_path(self) -> Path:
        return self.directory / REFERENCE_NAME

    @property
    def metadata_path(self) -> Path:
        return self.directory / METADATA_NAME

    def missing_files(self) -> Tuple[str, ...]:
        return tuple(name for name, path in (
            (VERIFIER_NAME, self.verifier_path),
            (REFERENCE_NAME, self.reference_path),
            (METADATA_NAME, self.metadata_path),
        ) if not path.is_file())

    def is_intact(self) -> bool:
        return not self.missing_files()


def default_store_root() -> Path:
    from utils.paths import get_paths

    return get_paths().root / "models" / "custom"


class CustomWakeWordStore:
    """The directory of enrolled wake words, and the only way in or out.

    Every path this returns is built from a validated slug under ``root``, so
    nothing a caller passes -- or a metadata file contains -- can address a
    file outside the store.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self._root = Path(root) if root is not None else None

    @property
    def root(self) -> Path:
        # Resolved on access rather than in __init__ so a test that redirects
        # AARMA_HOME after constructing the store still gets its own directory.
        return self._root if self._root is not None else default_store_root()

    # ------------------------------------------------------------ addressing

    def directory_for(self, phrase: str) -> Path:
        slug = slug_for_phrase(phrase)
        if not is_safe_slug(slug):
            raise CustomWakeWordError(f"{phrase!r} does not make a usable folder name")
        directory = (self.root / slug).resolve()
        root = self.root.resolve()
        # Belt and braces: the slug pattern already forbids separators, but a
        # path that escapes the store is the one failure worth being sure about.
        if root != directory and root not in directory.parents:
            raise CustomWakeWordError("refusing a model directory outside the store")
        return directory

    # --------------------------------------------------------------- reading

    def get(self, phrase: str) -> Optional[CustomWakeWord]:
        """The enrolled model for ``phrase``, or ``None``.

        Returns ``None`` rather than raising for anything wrong on disk -- a
        damaged model means "no detector for this identity", which the caller
        already knows how to report.
        """
        try:
            directory = self.directory_for(phrase)
        except CustomWakeWordError:
            return None
        return self._read(directory)

    def list(self) -> Tuple[CustomWakeWord, ...]:
        """Every intact enrolled model, in phrase order."""
        root = self.root
        if not root.is_dir():
            return ()
        found = []
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or not is_safe_slug(entry.name):
                continue
            model = self._read(entry)
            if model is not None and model.is_intact():
                found.append(model)
        return tuple(sorted(found, key=lambda m: m.phrase))

    def _read(self, directory: Path) -> Optional[CustomWakeWord]:
        meta_path = directory / METADATA_NAME
        if not meta_path.is_file():
            return None
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Unreadable custom wake-word metadata", extra={
                "context": {"path": str(meta_path), "error": str(exc)}})
            return None
        if not isinstance(raw, dict):
            return None
        if int(raw.get("schema_version", 0)) != SCHEMA_VERSION:
            logger.warning("Custom wake word has an unsupported layout", extra={
                "context": {"path": str(meta_path),
                            "schema_version": raw.get("schema_version")}})
            return None

        phrase = normalise_wake_phrase(raw.get("wake_phrase", ""))
        if not phrase or slug_for_phrase(phrase) != directory.name:
            # The folder name is authoritative; metadata that disagrees with it
            # is either corrupt or an attempt to answer for another phrase.
            logger.warning("Custom wake word does not match its folder", extra={
                "context": {"folder": directory.name, "phrase": phrase}})
            return None
        try:
            stage1 = float(raw["stage1_threshold"])
            stage2 = float(raw["stage2_threshold"])
        except (KeyError, TypeError, ValueError):
            logger.warning("Custom wake word has no usable thresholds", extra={
                "context": {"folder": directory.name}})
            return None
        if not (0.0 < stage1 <= 1.0) or not (0.0 < stage2 <= 1.0):
            logger.warning("Custom wake-word thresholds out of range", extra={
                "context": {"folder": directory.name,
                            "stage1": stage1, "stage2": stage2}})
            return None

        return CustomWakeWord(
            slug=directory.name,
            directory=directory,
            phrase=phrase,
            display_name=str(raw.get("display_name") or "").strip(),
            stage1_threshold=stage1,
            stage2_threshold=stage2,
            model_version=int(raw.get("model_version", 1)),
            created_at=str(raw.get("created_at", "")),
            enrollment_count=int(raw.get("enrollment_count", 0)),
            detector_type=str(raw.get("detector_type", DETECTOR_TYPE)),
            validation=dict(raw.get("validation") or {}),
            provenance=dict(raw.get("provenance") or {}),
            architecture=dict(raw.get("architecture") or {}),
        )

    # --------------------------------------------------------------- writing

    def install(self, *, phrase: str, display_name: str, verifier: Path,
                reference: Path, stage1_threshold: float, stage2_threshold: float,
                enrollment_count: int, validation: Dict[str, Any],
                provenance: Dict[str, Any],
                architecture: Optional[Dict[str, Any]] = None) -> CustomWakeWord:
        """Publish a validated model, replacing any previous one atomically.

        The new files are staged in a sibling directory and only swapped in once
        all three are written.  A previously working detector is therefore never
        destroyed by a failed install: either the whole new version appears, or
        the old one stays exactly as it was.
        """
        directory = self.directory_for(phrase)
        normalised = normalise_wake_phrase(phrase)
        verifier, reference = Path(verifier), Path(reference)
        for source in (verifier, reference):
            if not source.is_file():
                raise CustomWakeWordError(f"missing enrollment artefact: {source}")
        validate_checkpoint(verifier)

        previous = self._read(directory)
        version = (previous.model_version + 1) if previous is not None else 1

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "display_name": display_name,
            "normalised_name": display_name.strip().lower(),
            "wake_phrase": normalised,
            "slug": directory.name,
            "detector_type": DETECTOR_TYPE,
            "model_version": version,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "enrollment_count": int(enrollment_count),
            "stage1_threshold": round(float(stage1_threshold), 6),
            "stage2_threshold": round(float(stage2_threshold), 6),
            "validation": validation,
            "provenance": provenance,
            "architecture": architecture or {},
        }

        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}-new-",
                                        dir=str(self.root)))
        try:
            shutil.copyfile(verifier, staging / VERIFIER_NAME)
            shutil.copyfile(reference, staging / REFERENCE_NAME)
            (staging / METADATA_NAME).write_text(
                json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

            retired = directory.with_name(f".{directory.name}-old-{os.getpid()}")
            if directory.exists():
                os.replace(directory, retired)
            try:
                os.replace(staging, directory)
            except OSError:
                if retired.exists():          # put the working model back
                    os.replace(retired, directory)
                raise
            shutil.rmtree(retired, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(staging, ignore_errors=True)
            raise CustomWakeWordError(f"could not install the model: {exc}") from exc

        model = self._read(directory)
        if model is None:
            raise CustomWakeWordError("the installed model could not be read back")
        logger.info("Custom wake word installed", extra={"context": {
            "phrase": model.phrase, "version": model.model_version,
            "stage2_threshold": model.stage2_threshold}})
        return model

    def remove(self, phrase: str) -> bool:
        """Delete an enrolled model.  Returns whether anything was there."""
        directory = self.directory_for(phrase)
        if not directory.is_dir():
            return False
        shutil.rmtree(directory, ignore_errors=False)
        logger.info("Custom wake word removed", extra={
            "context": {"phrase": normalise_wake_phrase(phrase)}})
        return True


def validate_checkpoint(path: Path) -> Dict[str, Any]:
    """Read a stage-2 checkpoint safely and check it is the right shape.

    ``weights_only=True`` is the point of this function: it makes ``torch.load``
    refuse anything but tensors and plain data, so a checkpoint -- which is a
    file the user's own tooling wrote, but still a file -- cannot execute code
    on load.  The shape checks then reject a file that is well-formed but is
    not this architecture, before the caller tries to run it.

    Returns the checkpoint's metadata (``frames``, ``n_mels``, ``params``).
    """
    import torch

    path = Path(path)
    if not path.is_file():
        raise CustomWakeWordError(f"checkpoint missing: {path}")
    size = path.stat().st_size
    if size > MAX_CHECKPOINT_BYTES:
        raise CustomWakeWordError(
            f"checkpoint is {size / 1048576:.1f} MB; the verifier is a few "
            "hundred kilobytes, so this is not one")

    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise CustomWakeWordError(f"checkpoint could not be read safely: {exc}") from exc

    if not isinstance(blob, dict) or "state_dict" not in blob:
        raise CustomWakeWordError("checkpoint has no state_dict")
    state = blob["state_dict"]
    if not isinstance(state, dict) or not state:
        raise CustomWakeWordError("checkpoint state_dict is empty or malformed")

    total = 0
    for name, tensor in state.items():
        if not isinstance(name, str):
            raise CustomWakeWordError("checkpoint has a non-string parameter name")
        if not torch.is_tensor(tensor):
            raise CustomWakeWordError(f"checkpoint entry {name!r} is not a tensor")
        if tensor.dtype not in (torch.float32, torch.float64, torch.int64):
            raise CustomWakeWordError(
                f"checkpoint entry {name!r} has unexpected dtype {tensor.dtype}")
        total += tensor.numel()
    if total > MAX_PARAMETERS:
        raise CustomWakeWordError(f"checkpoint has {total:,} parameters; too large")

    n_mels = int(blob.get("n_mels", EXPECTED_MELS))
    frames = int(blob.get("frames", 0))
    if n_mels != EXPECTED_MELS:
        raise CustomWakeWordError(
            f"checkpoint expects {n_mels} mel bands, the front end produces "
            f"{EXPECTED_MELS}")
    if not (MIN_FRAMES <= frames <= MAX_FRAMES):
        raise CustomWakeWordError(f"checkpoint frame count {frames} is out of range")

    return {"frames": frames, "n_mels": n_mels, "parameters": total,
            "bytes": size}


def validate_reference(path: Path) -> Dict[str, Any]:
    """Check a stage-1 enrollment reference is a plain float embedding matrix."""
    import numpy as np

    path = Path(path)
    if not path.is_file():
        raise CustomWakeWordError(f"reference missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as blob:
            if "embeddings" not in blob:
                raise CustomWakeWordError("reference has no 'embeddings' array")
            embeddings = blob["embeddings"]
            shape = tuple(int(n) for n in embeddings.shape)
            dtype = str(embeddings.dtype)
    except CustomWakeWordError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CustomWakeWordError(f"reference could not be read: {exc}") from exc

    if len(shape) != 2 or shape[0] < 1 or shape[1] < 16:
        raise CustomWakeWordError(f"reference has an unusable shape {shape}")
    if not dtype.startswith("float"):
        raise CustomWakeWordError(f"reference is {dtype}, expected float")
    return {"shape": shape, "dtype": dtype}


#: The process-wide store.  A single instance so the registry, the enrollment
#: workflow and the settings screen all see the same directory.
STORE = CustomWakeWordStore()
