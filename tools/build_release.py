"""Build the FieldSOAR ZIP and embed a checksum manifest."""
import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", ".venv", ".venv_uavnetsim", "__pycache__", ".pytest_cache", ".mypy_cache", "reproduced"}
EXCLUDED_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".pyc", ".log", ".out", ".err")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error("Place the output ZIP outside the release tree")
    files = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if set(rel.parts) & EXCLUDED_DIRS or rel.name == "SHA256SUMS":
            continue
        if path.is_symlink():
            raise ValueError(f"Refusing to package a symbolic link: {rel}")
        if path.is_file() and not path.name.lower().endswith(EXCLUDED_SUFFIXES):
            files.append((rel.as_posix(), path))
    sums = []
    with ZipFile(output, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, path in files:
            payload = path.read_bytes()
            sums.append(f"{hashlib.sha256(payload).hexdigest()}  {name}\n")
            entry = ZipInfo(f"FieldSOAR-reproducibility/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, payload, compress_type=ZIP_DEFLATED, compresslevel=9)
        entry = ZipInfo("FieldSOAR-reproducibility/SHA256SUMS", date_time=(1980, 1, 1, 0, 0, 0))
        entry.create_system = 3
        entry.external_attr = 0o100644 << 16
        archive.writestr(entry, "".join(sums), compress_type=ZIP_DEFLATED, compresslevel=9)
    print(f"Wrote {output} with {len(files)} files")


if __name__ == "__main__":
    main()
