"""Create and verify a review ZIP from the allowlisted source manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--zip", dest="archive", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate.resolve(strict=True)
    archive = args.archive.resolve()
    if archive.exists():
        raise SystemExit(f"Refusing to overwrite ZIP: {archive}")
    manifest = candidate / "SOURCE_MANIFEST.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {row["release_path"]: row["release_sha256"] for row in rows}
    expected["SOURCE_MANIFEST.csv"] = sha256_bytes(manifest.read_bytes())
    with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=6) as output:
        for rel in sorted(expected):
            path = candidate / rel
            if not path.is_file():
                raise SystemExit(f"Missing allowlisted file: {rel}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected[rel]:
                raise SystemExit(f"Changed allowlisted file: {rel}")
            output.write(path, f"reproducibility_materials/{rel}")
    with ZipFile(archive) as assembled:
        names = assembled.namelist()
        expected_names = [f"reproducibility_materials/{rel}" for rel in sorted(expected)]
        if names != expected_names:
            raise SystemExit("ZIP member list differs from the release allowlist")
        for rel in expected:
            actual = sha256_bytes(assembled.read(f"reproducibility_materials/{rel}"))
            if actual != expected[rel]:
                raise SystemExit(f"ZIP member SHA-256 mismatch: {rel}")
    print(f"Verified ZIP: {archive}")
    print(f"Members: {len(expected)}")
    print(f"Bytes: {archive.stat().st_size}")
    print(f"SHA-256: {hashlib.sha256(archive.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
