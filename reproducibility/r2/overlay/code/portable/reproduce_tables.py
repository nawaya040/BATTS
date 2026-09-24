"""Reaggregate technical MSE tables from the bundled 50-seed records.

Usage: python code/portable/reproduce_tables.py <new-output-directory>
No estimator is fitted, and supplied results are read-only.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("GB", "FS", "BAT", "DRT (AdaBoost)", "CDC (AdaBoost)", "KLIEP", "uLSIF")
CELLS_2D = tuple((scenario, n0, n1) for scenario in
                 ("global_shift", "local_shift", "local_dispersion") for n0, n1 in
                 ((5000, 5000), (9000, 1000)))
CELLS_20D = tuple((scenario, n0, n1) for scenario in
                  ("global_shift", "latent_location_shift", "latent_dispersion") for n0, n1 in
                  ((5000, 5000), (9000, 1000)))


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream)


def mean_se(values: list[float]) -> tuple[float, float]:
    if len(values) < 2 or not all(map(math.isfinite, values)):
        raise ValueError("Mean/SE requires at least two finite values")
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(len(values))


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    if out.exists():
        raise SystemExit("Refusing to overwrite an existing output directory")
    r1 = ROOT / "reference/r1/output/section41_2d/full/table_generated/table1_summary.csv"
    table2d = ROOT / "results/r2/tables_2d"
    submitted = table2d / "submitted/per_seed_metrics.csv"
    stable = table2d / "stable/per_seed_metrics.csv"
    legacy_values: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
    for row in rows(r1):
        key = (row["method"], row["scenario"], int(row["n0"]), int(row["n1"]))
        legacy_values[key].append(float(row["mean_group_error"]))
    new_values: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
    finite_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    for row in rows(submitted):
        if row["table_id"] != "table1":
            continue
        cell = (row["scenario"], int(row["n0"]), int(row["n1"]))
        new_values[("DRT (AdaBoost)", *cell)].append(float(row["drt_mse_symmetric"]))
        value = float(row["cdc_mse_symmetric"] or "inf")
        if math.isfinite(value):
            new_values[("CDC (AdaBoost)", *cell)].append(value)
            finite_counts[cell] += 1
    table1: list[dict[str, object]] = []
    for method in METHODS:
        for scenario, n0, n1 in CELLS_2D:
            values = (new_values if method in ("DRT (AdaBoost)", "CDC (AdaBoost)")
                      else legacy_values)[(method, scenario, n0, n1)]
            expected = finite_counts[(scenario, n0, n1)] if method == "CDC (AdaBoost)" else 50
            if len(values) != expected:
                raise ValueError(f"Unexpected seed count: {method}/{scenario}/{n0}/{n1}")
            mean, se = mean_se(values)
            table1.append(dict(method=method, scenario=scenario, n0=n0, n1=n1,
                               finite_seeds=len(values), mean_mse=mean, mcse=se,
                               manuscript_mean=f"{mean:.3f}", manuscript_se=f"{se:.3f}"))

    stable_values: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in rows(stable):
        if row["table_id"] == "table1":
            key = (row["scenario"], int(row["n0"]), int(row["n1"]))
            stable_values[key].append(float(row["cdc_mse_symmetric"]))
    diagnostics: list[dict[str, object]] = []
    for cell in CELLS_2D:
        if finite_counts[cell] == 50:
            continue
        values = stable_values[cell]
        if len(values) != 50:
            raise ValueError(f"Stable CDC has fewer than 50 seeds: {cell}")
        mean, se = mean_se(values)
        diagnostics.append(dict(scenario=cell[0], n0=cell[1], n1=cell[2],
                                submitted_finite_seeds=finite_counts[cell],
                                stable_seeds=50, stable_mean_mse=mean, stable_mcse=se,
                                manuscript_mean=f"{mean:.3f}", manuscript_se=f"{se:.3f}"))

    mse_path = ROOT / "results/r2/summaries/revision-mse-summary-20260903T170144JST/mse_primary_by_seed.csv"
    grouped: dict[tuple[str, str, str, int, int], list[float]] = defaultdict(list)
    for row in rows(mse_path):
        scenario = row["scenario"]
        method = row["method"]
        source = row["source"]
        if scenario in ("global_shift", "null"):
            expected_source = ("canonical_bart_seed_metrics" if method == "BAT"
                               else "canonical_new_comparator_csv")
        else:
            expected_source = ("boosting_selection_rds" if method in
                               ("DRT (AdaBoost)", "CDC (AdaBoost)") else "legacy_20d_rds")
        if source != expected_source or scenario == "null":
            continue
        key = (row["representation"], method, scenario, int(row["n0"]), int(row["n1"]))
        value = float(row["mse_symmetric"] or "inf")
        grouped[key].append(value)
    tables20d: dict[str, list[dict[str, object]]] = {"raw": [], "transformed": []}
    for representation in tables20d:
        for method in METHODS:
            for scenario, n0, n1 in CELLS_20D:
                values = grouped[(representation, method, scenario, n0, n1)]
                if len(values) != 50:
                    raise ValueError(f"Unexpected 20D seed count: {representation}/{method}/{scenario}/{n0}/{n1}")
                mean, se = mean_se(values)
                tables20d[representation].append(dict(
                    method=method, scenario=scenario, n0=n0, n1=n1,
                    seeds=50, mean_mse=mean, mcse=se,
                    manuscript_mean=f"{mean:.3f}", manuscript_se=f"{se:.3f}"))

    out.mkdir(parents=True)
    write_csv(out / "main_table1_2d.csv", table1)
    write_csv(out / "supp_cdc_diagnostics.csv", diagnostics)
    write_csv(out / "main_table2_20d.csv", tables20d["raw"])
    write_csv(out / "supp_transformed_20d.csv", tables20d["transformed"])
    print(f"Wrote four technical table files to {out}")


if __name__ == "__main__":
    main()
