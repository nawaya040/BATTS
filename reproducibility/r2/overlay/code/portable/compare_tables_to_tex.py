"""Compare rounded technical tables with the final manuscript TeX sources.

Usage: python code/portable/compare_tables_to_tex.py MAIN_TEX SUPPLEMENT_TEX TABLE_OUTPUT_DIR
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


METHODS = ("GB", "FS", "BAT", "DRT (AdaBoost)", "CDC (AdaBoost)", "KLIEP", "uLSIF")
SPECS = (
    ("main_table1_2d.csv", "main", "table: MSE(2D)"),
    ("main_table2_20d.csv", "main", "table: MSE(multi)"),
    ("supp_transformed_20d.csv", "supp", "table: MSE(multi, alt)"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def tex_rows(tex: str, label: str) -> dict[str, tuple[list[str], list[str]]]:
    start = tex.index(r"\label{" + label + "}")
    a = tex.index(r"\begin{tabular}", start)
    b = tex.index(r"\end{tabular}", a)
    lines = tex[a:b].splitlines()
    found: dict[str, tuple[list[str], list[str]]] = {}
    for index, line in enumerate(lines[:-1]):
        for method in METHODS:
            if re.match(r"^\s*" + re.escape(method) + r"\s*&", line):
                found[method] = (
                    re.findall(r"\d+\.\d+", line),
                    re.findall(r"\d+\.\d+", lines[index + 1]),
                )
    if set(found) != set(METHODS):
        raise ValueError(f"Incomplete TeX table: {label}")
    return found


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main_tex = Path(sys.argv[1]).read_text(encoding="utf-8")
    supp_tex = Path(sys.argv[2]).read_text(encoding="utf-8")
    output = Path(sys.argv[3])
    mismatches: list[str] = []
    compared = 0
    for filename, source, label in SPECS:
        observed = read_csv(output / filename)
        reported = tex_rows(main_tex if source == "main" else supp_tex, label)
        for method in METHODS:
            selected = [row for row in observed if row["method"] == method]
            if len(selected) != 6:
                raise ValueError(f"Expected six cells: {filename}/{method}")
            for kind, expected in zip(("manuscript_mean", "manuscript_se"), reported[method]):
                actual = [row[kind] for row in selected]
                compared += len(actual)
                if actual != expected:
                    mismatches.append(f"{filename}/{method}/{kind}: {actual} versus {expected}")

    block = supp_tex[supp_tex.index(r"\label{table: cdc numerical diagnostics}"):]
    block = block[:block.index(r"\end{tabular}")]
    diagnostics = re.findall(
        r"\\rnew\{(Local Shift|Local Dispersion)\}.*?"
        r"\\rnew\{(0\.5|0\.9)\}.*?\\rnew\{(\d+/50)\}.*?"
        r"\\rnew\{([\d.]+) \(([\d.]+)\)\}", block,
    )
    data = read_csv(output / "supp_cdc_diagnostics.csv")
    lookup = {
        (row["scenario"], "0.5" if row["n0"] == row["n1"] else "0.9"): row
        for row in data
    }
    if len(diagnostics) != 4:
        raise ValueError("Expected four CDC diagnostic rows")
    for scenario, balance, finite, mean, se in diagnostics:
        key = ("local_shift" if scenario == "Local Shift" else "local_dispersion", balance)
        row = lookup[key]
        actual = (row["submitted_finite_seeds"] + "/50", row["manuscript_mean"], row["manuscript_se"])
        compared += 3
        if actual != (finite, mean, se):
            mismatches.append(f"CDC diagnostics {key}: {actual} versus {(finite, mean, se)}")
    if mismatches:
        raise SystemExit("\n".join(mismatches))
    print(f"PASS: {compared} displayed numeric entries match the TeX sources")


if __name__ == "__main__":
    main()
