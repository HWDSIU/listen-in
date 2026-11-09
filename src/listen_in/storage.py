"""Output layout and safe writes for local paths and Databricks Volumes.

Layout under ``output_dir``::

    audio/  transcripts/  analysis/  manifests/

Work happens in a local staging directory (normally under the system temp dir, i.e.
``/tmp`` on a Databricks driver); finished artefacts are then copied to the
destination. Writes go to ``<name>.partial`` then ``os.replace``; if the destination
filesystem refuses the rename (some FUSE mounts), we fall back to a direct copy.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .exceptions import OutputExistsError
from .utils import validate_output_dir

log = logging.getLogger(__name__)

SUBDIRS = ("audio", "transcripts", "analysis", "manifests")


@dataclass(frozen=True)
class OutputLayout:
    root: Path

    @classmethod
    def create(cls, output_dir: str | Path) -> OutputLayout:
        root = validate_output_dir(output_dir)
        for sub in SUBDIRS:
            (root / sub).mkdir(parents=True, exist_ok=True)
        return cls(root)

    @property
    def audio(self) -> Path:
        return self.root / "audio"

    @property
    def transcripts(self) -> Path:
        return self.root / "transcripts"

    @property
    def analysis(self) -> Path:
        return self.root / "analysis"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"


def default_staging_dir() -> Path:
    return Path(tempfile.gettempdir()) / "listen-in" / "staging"


def _finalise(tmp: Path, dest: Path) -> None:
    try:
        os.replace(tmp, dest)
    except OSError:
        # Some mounted filesystems do not support rename-over; copy instead.
        shutil.copyfile(tmp, dest)
        tmp.unlink(missing_ok=True)


def _check_overwrite(dest: Path, overwrite: bool) -> None:
    if dest.exists() and not overwrite:
        raise OutputExistsError(f"{dest} already exists (pass overwrite=True to replace it).")


def write_text(dest: Path, text: str, *, overwrite: bool = False) -> Path:
    _check_overwrite(dest, overwrite)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    tmp.write_text(text, encoding="utf-8")
    _finalise(tmp, dest)
    return dest


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    return obj


def write_json(dest: Path, obj: Any, *, overwrite: bool = False) -> Path:
    text = json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2) + "\n"
    return write_text(dest, text, overwrite=overwrite)


def publish_file(src: Path, dest: Path, *, overwrite: bool = False, move: bool = False) -> Path:
    """Copy (or move) a finished file from staging to its destination."""
    _check_overwrite(dest, overwrite)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dest.resolve():
        return dest
    tmp = dest.with_name(dest.name + ".partial")
    shutil.copyfile(src, tmp)
    _finalise(tmp, dest)
    if move:
        src.unlink(missing_ok=True)
    return dest
