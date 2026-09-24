"""Add the verified raw global/null BART summary to an existing candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from shutil import copyfileobj

from sanitize_provenance import PATTERN, alias


FIELDS = (
    "release_path", "origin", "source_bytes", "source_sha256",
    "release_bytes", "release_sha256", "transformation",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate.resolve(strict=True)
    source = args.source.resolve(strict=True)
    manifest = candidate / "SOURCE_MANIFEST.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = {row["release_path"]: row for row in csv.DictReader(stream)}
    count = 0
    for item in sorted(source.iterdir()):
        if not item.is_file():
            continue
        rel = f"results/r2/summaries/raw-global-null-20260813T162558/{item.name}"
        if rel in rows:
            raise SystemExit(f"Already listed: {rel}")
        out = candidate / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(item)
        with item.open("rb") as inp, out.open("xb") as dst:
            copyfileobj(inp, dst, 1024 * 1024)
        if sha256(out) != before:
            raise SystemExit(f"Copy mismatch: {rel}")
        transformation = "none"
        if out.suffix.lower() in (".csv", ".txt", ".md"):
            raw = out.read_text(encoding="utf-8")
            clean, n = PATTERN.subn(alias, raw)
            if n:
                out.write_text(clean, encoding="utf-8", newline="")
                transformation = "machine_path_aliases"
        rows[rel] = {
            "release_path": rel,
            "origin": f"R2_RAW_BART_SUMMARY/{item.name}",
            "source_bytes": item.stat().st_size,
            "source_sha256": before,
            "release_bytes": out.stat().st_size,
            "release_sha256": sha256(out),
            "transformation": transformation,
        }
        count += 1
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))
    print(f"Added and verified {count} raw global/null summary files")


if __name__ == "__main__":
    main()
