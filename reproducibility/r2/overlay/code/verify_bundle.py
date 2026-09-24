"""Verify the release files and selected seed-level result invariants.

Run from the release root: python code/verify_bundle.py
This program reads inputs only and never changes scientific results.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream)


def main() -> None:
    source_manifest = ROOT / "SOURCE_MANIFEST.csv"
    listed = list(csv_rows(source_manifest))
    require(len(listed) >= 1200, "Source manifest is unexpectedly short")
    for row in listed:
        path = ROOT / row["release_path"]
        require(path.is_file(), f"Missing file: {row['release_path']}")
        require(path.stat().st_size == int(row["release_bytes"]),
                f"Byte-count mismatch: {row['release_path']}")
        require(sha256(path) == row["release_sha256"],
                f"SHA-256 mismatch: {row['release_path']}")

    summary = ROOT / "results/r2/summaries"
    mse = summary / "revision-mse-summary-20260903T170144JST/mse_primary_by_seed.csv"
    mse_seeds: dict[tuple[str, ...], set[int]] = defaultdict(set)
    mse_count = 0
    for row in csv_rows(mse):
        key = tuple(row[name] for name in (
            "method", "scenario", "n0", "n1", "representation", "selection"
        ))
        mse_seeds[key].add(int(row["seed"]))
        mse_count += 1
    require(mse_count > 1000, "MSE seed-level results are unexpectedly short")
    require(all(seeds == set(range(1, 51)) for seeds in mse_seeds.values()),
            "An MSE setting does not contain seeds 1–50")

    coverage = summary / "coverage-summary-20260903T170144JST/coverage_by_seed.csv"
    coverage_seeds: dict[tuple[str, ...], set[int]] = defaultdict(set)
    coverage_count = 0
    for row in csv_rows(coverage):
        key = tuple(row[name] for name in (
            "family", "scenario", "n0", "n1", "representation", "group"
        ))
        coverage_seeds[key].add(int(row["seed"]))
        coverage_count += 1
    require(coverage_count > 1000, "Coverage seed-level results are unexpectedly short")
    require(all(seeds == set(range(1, 51)) for seeds in coverage_seeds.values()),
            "A coverage setting does not contain seeds 1–50")

    detail = ROOT / "results/r2/figure_details"
    require(len(list((detail / "2d").rglob("*_details.rds"))) == 6,
            "Expected six 2D detailed results")
    require(len(list((detail / "20d").glob("*_details.rds"))) == 1,
            "Expected one 20D detailed result")
    checksums = list(csv_rows(detail / "boosting/output_checksums.csv"))
    checksum_by_job = {row["job_id"]: row for row in checksums}
    for path in (detail / "boosting").glob("*.rds"):
        job = path.stem
        require(job in checksum_by_job, f"Missing boosting checksum row: {job}")
        row = checksum_by_job[job]
        require(sha256(path) == row["output_sha256"].lower(),
                f"Boosting checksum mismatch: {job}")
        require(path.stat().st_size == int(row["output_bytes"]),
                f"Boosting byte-count mismatch: {job}")

    print(f"PASS: {len(listed)} release files match the source manifest")
    print(f"PASS: {len(mse_seeds)} MSE settings, {mse_count} seed-level rows")
    print(f"PASS: {len(coverage_seeds)} coverage settings, {coverage_count} seed-level rows")
    print("PASS: seven selected detailed RDS files and boosting checksums")


if __name__ == "__main__":
    main()
