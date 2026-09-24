"""Copy a named external result tree into a candidate with verified hashes."""

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
    parser.add_argument("--release-rel", required=True)
    parser.add_argument("--origin-label", required=True)
    args = parser.parse_args()
    candidate = args.candidate.resolve(strict=True)
    source = args.source.resolve(strict=True)
    rel_root = Path(args.release_rel)
    if rel_root.is_absolute() or ".." in rel_root.parts:
        raise SystemExit("--release-rel must stay within the candidate")
    manifest = candidate / "SOURCE_MANIFEST.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = {row["release_path"]: row for row in csv.DictReader(stream)}
    count = 0
    for item in sorted(source.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(source)
        rel = (rel_root / relative).as_posix()
        if rel in rows:
            raise SystemExit(f"Already listed: {rel}")
        out = candidate / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        original_hash = sha256(item)
        with item.open("rb") as inp, out.open("xb") as dst:
            copyfileobj(inp, dst, 1024 * 1024)
        if sha256(out) != original_hash:
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
            "origin": f"{args.origin_label}/{relative.as_posix()}",
            "source_bytes": item.stat().st_size,
            "source_sha256": original_hash,
            "release_bytes": out.stat().st_size,
            "release_sha256": sha256(out),
            "transformation": transformation,
        }
        count += 1
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))
    print(f"Added and verified {count} files under {rel_root.as_posix()}")


if __name__ == "__main__":
    main()
