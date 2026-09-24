"""Add English release overlay files to the source manifest, then verify it."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


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
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    candidate = args.candidate.resolve(strict=True)
    r2 = Path(__file__).resolve().parent
    overlay = r2 / "overlay"
    manifest = candidate / "SOURCE_MANIFEST.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = {row["release_path"]: row for row in csv.DictReader(stream)}
    for source in sorted(overlay.rglob("*")):
        if not source.is_file():
            continue
        rel = source.relative_to(overlay).as_posix()
        release = candidate / rel
        if not release.is_file():
            raise SystemExit(f"Overlay file missing in release: {rel}")
        source_hash = sha256(source)
        release_hash = sha256(release)
        if source_hash != release_hash:
            raise SystemExit(f"Overlay source differs from release: {rel}")
        rows[rel] = {
            "release_path": rel,
            "origin": f"R2_REPO/reproducibility/r2/overlay/{rel}",
            "source_bytes": source.stat().st_size,
            "source_sha256": source_hash,
            "release_bytes": release.stat().st_size,
            "release_sha256": release_hash,
            "transformation": "none",
        }
    all_files = {p.relative_to(candidate).as_posix() for p in candidate.rglob("*") if p.is_file()}
    # An R graphics startup can leave this scratch file after a failed render.
    # It is deliberately outside the release allowlist and excluded from ZIP.
    unlisted = all_files - set(rows) - {"SOURCE_MANIFEST.csv", "Rplots.pdf"}
    if unlisted:
        raise SystemExit(f"Release files missing from manifest: {sorted(unlisted)}")
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))
    print(f"Finalized manifest: {len(rows)} files")


if __name__ == "__main__":
    main()
