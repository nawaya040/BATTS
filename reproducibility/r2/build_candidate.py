"""Assemble the review candidate from explicitly supplied, read-only roots.

The command refuses to overwrite an existing candidate. No scientific
calculation is run. Source and destination SHA-256 hashes are recorded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from pathlib import Path
from shutil import copyfileobj


BATTS_COMMIT = "6f625bad83702b36e5480be1ed1343258a9b075a"
WORKFLOW_COMMIT = "f921525711faef2a326c561820749e0c70f1bf42"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repo-root", "r1-root", "legacy-root", "boosting-root",
        "boosting-checksums", "canonical-root", "raw-bart-summary-root",
        "cdc-submitted-root", "cdc-stable-root", "candidate-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve(strict=True)
    r1 = args.r1_root.resolve(strict=True)
    legacy = args.legacy_root.resolve(strict=True)
    boosting = args.boosting_root.resolve(strict=True)
    checksum = args.boosting_checksums.resolve(strict=True)
    canonical = args.canonical_root.resolve(strict=True)
    raw_bart = args.raw_bart_summary_root.resolve(strict=True)
    cdc_submitted = args.cdc_submitted_root.resolve(strict=True)
    cdc_stable = args.cdc_stable_root.resolve(strict=True)
    target = args.candidate_root.resolve()
    if target.exists():
        raise SystemExit(f"Candidate already exists; refusing overwrite: {target}")
    target.mkdir(parents=True)
    records: list[dict[str, str | int]] = []

    roots = (
        (r1, "R1_SUBMISSION"), (repo, "R2_REPO"),
        (legacy, "LEGACY_RESULTS"), (boosting, "BOOSTING_RESULTS"),
        (canonical, "R2_CANONICAL"), (raw_bart, "R2_RAW_BART_SUMMARY"),
        (cdc_submitted, "R2_2D_CDC_SUBMITTED"),
        (cdc_stable, "R2_2D_CDC_STABLE"),
    )

    def logical_origin(source: Path) -> str:
        for root, label in roots:
            if source.is_relative_to(root):
                return f"{label}/{source.relative_to(root).as_posix()}"
        if source == checksum:
            return "BOOSTING_CHECKSUMS/output_checksums.csv"
        raise RuntimeError(f"Source root not identified: {source}")

    def record(destination: Path, origin: str, source_hash: str) -> None:
        destination_hash = sha256(destination)
        if destination_hash != source_hash:
            raise RuntimeError(f"Copy hash mismatch: {destination}")
        records.append({
            "release_path": destination.relative_to(target).as_posix(),
            "origin": origin,
            "bytes": destination.stat().st_size,
            "sha256": destination_hash,
        })

    def copy(source: Path, rel: str) -> None:
        source = source.resolve(strict=True)
        destination = target / rel
        if destination.exists():
            raise RuntimeError(f"Duplicate release path: {rel}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(source)
        with source.open("rb") as inp, destination.open("xb") as out:
            copyfileobj(inp, out, 1024 * 1024)
        record(destination, logical_origin(source), before)

    def copy_tree(source: Path, rel: str) -> None:
        for item in sorted(source.rglob("*")):
            if item.is_file():
                copy(item, (Path(rel) / item.relative_to(source)).as_posix())

    # Preserve the previous submission as an identifiable reference. The
    # installed R1 r_lib is machine-specific and is intentionally omitted.
    for part in ("data", "output"):
        copy_tree(r1 / part, f"reference/r1/{part}")
    for part in ("methods", "scripts", "utils"):
        copy_tree(r1 / "code" / part, f"reference/r1/code/{part}")
    for item in (r1 / "code").iterdir():
        if item.is_file():
            copy(item, f"reference/r1/code/{item.name}")
    for item in r1.iterdir():
        if item.is_file() and item.suffix.lower() in (".md", ".txt"):
            copy(item, f"reference/r1/{item.name}")

    # Extract the exact BATTS Git tree, independently of the current worktree.
    names = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", BATTS_COMMIT],
        text=True,
    ).splitlines()
    for name in names:
        if name.startswith(".git") or name.startswith("src/.git"):
            continue
        payload = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{BATTS_COMMIT}:{name}"]
        )
        destination = target / "code" / "BATTS" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as out:
            out.write(payload)
        record(destination, f"git:{BATTS_COMMIT}:{name}", hashlib.sha256(payload).hexdigest())

    # Current approved R2 workflow and display scripts. The four scripts
    # containing machine-specific paths are excluded; portable release
    # entry points are supplied separately.
    excluded = {
        "scripts/figures/plot_figure3_r2_preview.R",
        "scripts/figures/plot_figure4_r2_preview.R",
        "scripts/figures/plot_supplement_s1_r2_preview.R",
        "scripts/figures/plot_supplement_s2_s3_r2_preview.R",
        "scripts/revision/REVISION_MANUSCRIPT_PLAN.md",
    }
    for item in sorted((repo / "scripts").rglob("*")):
        if item.is_file():
            rel = item.relative_to(repo).as_posix()
            if rel not in excluded:
                copy(item, f"code/r2/{rel}")

    # All compact seed-level summary files, with original directories kept.
    copy_tree(canonical / "summaries", "results/r2/summaries")
    copy_tree(raw_bart, "results/r2/summaries/raw-global-null-20260813T162558")
    copy_tree(cdc_submitted, "results/r2/tables_2d/submitted")
    copy_tree(cdc_stable, "results/r2/tables_2d/stable")

    # Figure 3 and Supplement S2/S3: six exact 2D settings.
    settings_2d = (
        ("local_shift", 5000, 5000, 1),
        ("global_shift", 5000, 5000, 1),
        ("local_dispersion", 5000, 5000, 21),
        ("global_shift", 9000, 1000, 1),
        ("local_shift", 9000, 1000, 16),
        ("local_dispersion", 9000, 1000, 16),
    )
    for scenario, n0, n1, seed in settings_2d:
        filename = f"{scenario}_{n0}_{n1}_5_{seed}_details.rds"
        copy(legacy / "experiments_2D" / scenario / filename,
             f"results/r2/figure_details/2d/{scenario}/{filename}")
        job = (f"boosting_selection_2d_{scenario}_n0-{n0}_n1-{n1}"
               f"_transformed-false_seed-{seed:03d}.rds")
        copy(boosting / "outputs" / "canonical" / job,
             f"results/r2/figure_details/boosting/{job}")

    # Supplement S5: exact 20D seed used for the plotted surface.
    s5 = "latent_location_shift_5000_5000_5_21_details.rds"
    copy(legacy / "experiments_multi" / "latent_location_shift" / s5,
         f"results/r2/figure_details/20d/{s5}")
    s5_job = ("boosting_selection_20d_latent_location_shift_n0-5000_n1-5000"
              "_transformed-false_seed-021.rds")
    copy(boosting / "outputs" / "canonical" / s5_job,
         f"results/r2/figure_details/boosting/{s5_job}")
    copy(checksum, "results/r2/figure_details/boosting/output_checksums.csv")

    # Supplement S1: saved 50-repeat input series.
    copy_tree(legacy / "experiments_1D" / "boosting",
              "results/r2/supplement_s1/boosting")

    # Figure 5: the exact saved plotted series and its provenance table.
    for suffix in ("summary.csv", "by_repeat.csv", "input_manifest.csv"):
        filename = f"figure_20d_localization_r2_preview_{suffix}"
        copy(repo / "output" / "figures" / filename,
             f"results/r2/figure5/{filename}")

    overlay = repo / "reproducibility" / "r2" / "overlay"
    if overlay.exists():
        copy_tree(overlay, "")

    manifest = target / "SOURCE_MANIFEST.csv"
    with manifest.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("release_path", "origin", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: str(row["release_path"])))
    print(f"Candidate: {target}")
    print(f"Verified files: {len(records)}")
    print(f"Total bytes: {sum(int(row['bytes']) for row in records)}")
    print(f"BATTS commit: {BATTS_COMMIT}")
    print(f"R2 workflow source commit: {WORKFLOW_COMMIT}")


if __name__ == "__main__":
    main()
