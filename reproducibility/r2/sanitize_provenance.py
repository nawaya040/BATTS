"""Replace machine-local paths in copied text provenance, retaining hashes.

Scientific numeric CSV columns and RDS files are never modified. The source
SHA-256 remains in SOURCE_MANIFEST.csv; release hashes are recorded separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path


PATTERN = re.compile(r'[A-Za-z]:/Users/[^/\s,"\r\n]+/[^,"\s\r\n]+')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alias(match: re.Match[str]) -> str:
    original = match.group(0)
    key = hashlib.sha256(original.encode("utf-8")).hexdigest()[:12]
    return f"ORIGINAL_LOCATION_{key}/{original.rsplit('/', 1)[-1]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    root = args.candidate.resolve(strict=True)
    manifest = root / "SOURCE_MANIFEST.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    changed: set[str] = set()
    for row in rows:
        rel = str(row["release_path"])
        path = root / rel
        if path.suffix.lower() in (".csv", ".txt", ".md"):
            raw = path.read_text(encoding="utf-8")
            clean, count = PATTERN.subn(alias, raw)
            if count:
                path.write_text(clean, encoding="utf-8", newline="")
                changed.add(rel)
        row["source_sha256"] = row.pop("sha256")
        row["source_bytes"] = row.pop("bytes")
        row["release_bytes"] = path.stat().st_size
        row["release_sha256"] = digest(path)
        row["transformation"] = "machine_path_aliases" if rel in changed else "none"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "release_path", "origin", "source_bytes", "source_sha256",
            "release_bytes", "release_sha256", "transformation",
        ))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Aliased local paths in {len(changed)} provenance files")


if __name__ == "__main__":
    main()
