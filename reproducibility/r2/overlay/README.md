# JASA Round 2 reproducibility materials — review candidate

This directory is a review candidate for the technical part of the Round 2 revision. It contains the submitted Round 1 reference materials, the exact BATTS source revision specified by the Round 2 computation scripts, the Round 2 workflow scripts, selected saved results, and portable drawing and light-run entry points. Section 5 requires coauthor completion; see [COAUTHOR_SECTION5.md](COAUTHOR_SECTION5.md). This candidate has not yet been certified against every final manuscript table and figure.

## What the saved results permit

The `results/r2/summaries/` directories contain seed-level results for the Round 2 MSE, coverage, raw global/null and transformed BART calibration, and comparator analyses. They support reinspection of the 50-repeat summaries without rerunning every estimator. `results/r2/tables_2d/` contains the submitted and stable CDC seed-level summaries used for Main Table 1 and the supplementary diagnostic table. `results/r2/figure_details/` contains the selected 2D and 20D detailed RDS files used for the displayed estimation surfaces, together with the matching boosting RDS files and checksum table. `results/r2/supplement_s1/` contains the fixed-tree AdaBoost/GB results for Supplementary Figure S1. The complete 20D transformed-BART RDS set is approximately 14 GB and is represented here by its seed-level summaries, source code, and provenance. Full recomputation is a separate, expensive procedure.

The `code/light/run_light.R` entry point runs one selected simulation setting with the same data seed as a displayed figure and the archived Round 1 light hyperparameters. Light estimates may differ visibly and numerically from the manuscript. Its output is marked `SMOKE_NOT_FOR_PAPER`. It must not be substituted for the saved 50-repeat results.

## Directory guide

| Directory | Contents |
|---|---|
| `reference/r1/` | Prior submitted code, processed Section 5 data, and saved results. The machine-specific R library was omitted. |
| `code/BATTS/` | BATTS source at Git commit `6f625bad83702b36e5480be1ed1343258a9b075a`. |
| `code/r2/scripts/` | Round 2 calculation, summary, and drawing scripts. Scripts with fixed machine paths were excluded from this snapshot. |
| `code/portable/` | Release-specific drawing copies with explicit input/output arguments. |
| `code/light/` | Seed-matched reduced-computation demonstration. |
| `results/r2/` | Saved seed-level summaries, selected detailed results, and hashes. |
| `SOURCE_MANIFEST.csv` | Origin, original SHA-256, release SHA-256, and any path-only redaction for every assembled source file. |

Local absolute paths in copied provenance text/CSV files were replaced with stable aliases. The original file hash and release hash are both recorded in `SOURCE_MANIFEST.csv`. Numerical summary values and RDS contents were not altered.

## Environment and safe output

The original submitted workflow used R 4.5.2. R packages are not bundled as a binary library; install the supplied source packages and required CRAN packages in a local library, using a compiler toolchain for packages with C++ source. See [ENVIRONMENT.md](ENVIRONMENT.md). Run commands from this directory. Replace `<new-output-dir>` with an empty path outside this package. The selected drawing scripts reject an existing output directory. Preserve the supplied `reference/` and `results/` trees unchanged.

## Selected figures from saved inputs

These commands describe the drawing paths. Their execution status on this candidate is recorded in [VALIDATION_STATUS.md](VALIDATION_STATUS.md).

```sh
Rscript code/r2/scripts/figures/plot_figure1_r2_preview.R --results-root=reference/r1/output/section34_1d/full --png-output=<new-output-dir>/figure1_main.png --pdf-output=<new-output-dir>/figure1_main.pdf --setting-preset=balanced-sample-size

Rscript code/r2/scripts/figures/plot_figure1_r2_preview.R --results-root=reference/r1/output/section34_1d/full --png-output=<new-output-dir>/figure1_supp.png --pdf-output=<new-output-dir>/figure1_supp.pdf --setting-preset=unbalanced-sample-size

Rscript code/r2/scripts/figures/plot_figure2_r2_preview.R --png-output=<new-output-dir>/figure2.png --pdf-output=<new-output-dir>/figure2.pdf

Rscript code/r2/scripts/figures/plot_figure3_compact_preview.R --legacy-detail=results/r2/figure_details/2d/local_shift/local_shift_5000_5000_5_1_details.rds --boosting-result=results/r2/figure_details/boosting/boosting_selection_2d_local_shift_n0-5000_n1-5000_transformed-false_seed-001.rds --boosting-checksums=results/r2/figure_details/boosting/output_checksums.csv --output-dir=<new-output-dir>

Rscript code/r2/scripts/figures/plot_figure4_compact_preview.R results/r2/summaries/raw-global-null-20260813T162558 results/r2/summaries/coverage-summary-20260903T170144JST <new-output-dir>

Rscript code/portable/plot_figure5.R results/r2/figure5/figure_20d_localization_r2_preview_summary.csv <new-output-dir>

Rscript code/portable/plot_supplement_s1.R --input-root=results/r2/supplement_s1/boosting --output-dir=<new-output-dir>

Rscript code/r2/scripts/figures/plot_supplement_eight_panel_release.R results/r2/figure_details/2d results/r2/figure_details/20d/latent_location_shift_5000_5000_5_21_details.rds results/r2/figure_details/boosting results/r2/figure_details/boosting/output_checksums.csv <new-output-dir>

Rscript code/r2/scripts/figures/plot_eight_panel_equal_size_release.R <Figure-3-output-dir>/plotted_data.rds <supplement-surface-output-dir> <new-output-dir>

Rscript code/portable/plot_supplement_s4.R results/r2/summaries/coverage-summary-20260903T170144JST <new-output-dir>

Rscript code/r2/scripts/figures/plot_supplement_s6_compact_preview.R <new-output-dir>
```

Run each command with a distinct output directory. The first surface commands create `plotted_data.rds`; the equal-size command generates the final Figure 3 and S2/S3/S5 page geometry from those saved plot data objects. The seven resulting PDFs were checked against the final manuscript assets at 110 dpi and had identical pixels. Other validated drawing commands and the remaining Figure 2 layout difference are recorded in [VALIDATION_STATUS.md](VALIDATION_STATUS.md).

The technical tables can be regenerated from bundled seed-level records without fitting estimators:

```sh
python code/portable/reproduce_tables.py <new-output-dir>
```

This writes Main Tables 1 and 2, the transformed 20D table, and the CDC diagnostic table as CSV. The displayed values and standard errors were compared cell by cell with the final local TeX and matched after three-decimal rounding. The Round 1 Table 1 script and older values remain under `reference/r1/` for historical comparison.

## Reduced runs at the displayed seeds

The following example exercises the 2D local-shift setting used for Main Figure 3. The supplementary 2D settings and their seeds are listed in [FIGURE_TABLE_MAP.md](FIGURE_TABLE_MAP.md). Change the setting arguments and use a fresh output directory for each run. The 20D balanced latent-location S5 setting uses `--family=20d --scenario=latent_location_shift --n0=5000 --n1=5000 --seed=21`.

```sh
Rscript code/light/run_light.R --family=2d --scenario=local_shift --n0=5000 --n1=5000 --seed=1 --lib-dir=<installed-R-library> --output-dir=<new-output-dir>
```

The selected R1 light parameters use 40 Bayesian trees, 40 burn-in sweeps, and 80 backfitting sweeps; the full submitted settings use 200 trees, 2,000 burn-in sweeps, and 1,000 backfitting sweeps. The 2D light setting also reduces cross-validation and boosting work. Data seeds match the displayed settings, while the lower computation budget can alter all estimated values.

## Manuscript and application handoff

[FIGURE_TABLE_MAP.md](FIGURE_TABLE_MAP.md) identifies the saved input for every technical table and figure and distinguishes confirmed inputs from outstanding manuscript comparisons. [COAUTHOR_SECTION5.md](COAUTHOR_SECTION5.md) is the author-facing placeholder for the application section. The Round 1 processed data and outputs are present as references; the coauthor should identify which Round 2 plots and data versions are final, provide the missing rendering commands, and confirm data-sharing terms before this package is submitted.
