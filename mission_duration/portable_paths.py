"""Resolve archived project paths in a relocated checkout without changing the source manifests."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_PREFIXES = tuple(map(Path, (
    "/mnt/pixstor/data/ka9d9/fieldvision",
    "/home/ka9d9/data/fieldvision",
)))
NativePath = type(Path())


def project_path(value):
    path = Path(value)
    for prefix in ORIGINAL_PREFIXES:
        if path.is_relative_to(prefix):
            return ROOT / path.relative_to(prefix)
    return path


class ArchivedPath(NativePath):
    """Translate project-root paths passed to the frozen modules' Path constructors."""
    def __new__(cls, *parts, **kwargs):
        return super().__new__(cls, os.fspath(project_path(Path(*parts))), **kwargs)

    def __init__(self, *parts, **kwargs):
        if hasattr(NativePath, "with_segments"):  # pathlib initializes paths in __init__ on Python 3.12+
            super().__init__(os.fspath(project_path(Path(*parts))), **kwargs)
