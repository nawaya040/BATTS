# Round 2 technical output map

This map names the saved inputs and drawing code included in this candidate. A listed input is not a claim that the final manuscript asset has already been matched byte for byte. The full 50-repeat results are read from saved RDS or seed-level CSV files. A light run checks one selected computational path and is labeled separately.

| Manuscript result | Included source and saved input | Current verification boundary |
|---|---|---|
| Main Figure 1, one-dimensional posterior and coverage | `reference/r1/output/section34_1d/full/`; `code/r2/scripts/figures/plot_figure1_r2_preview.R` with `balanced-sample-size` preset | Rendered PDF matches the final manuscript asset pixel for pixel at 110 dpi. |
| Main Figure 2, two-dimensional generated data | `code/r2/scripts/figures/plot_figure2_r2_preview.R` and `code/r2/scripts/coverage/models/section41_2d_models.R` | Fixed seed 1 is in the code. The regenerated PDF uses the same plotted scenarios, but panel placement differs from the final PDF; 17.3% of pixels differ at 110 dpi. |
| Main Table 1, two-dimensional MSE/SE | `reference/r1/output/section41_2d/full/table_generated/table1_summary.csv`; `results/r2/tables_2d/submitted/per_seed_metrics.csv`; `code/portable/reproduce_tables.py` | All 42 displayed means and 42 SEs match final TeX after rounding. |
| Main Figure 3, two-dimensional local-shift estimation surface | `results/r2/figure_details/2d/local_shift/local_shift_5000_5000_5_1_details.rds`; matching seed-001 boosting RDS; `plot_figure3_compact_preview.R` followed by `plot_eight_panel_equal_size_release.R` | Rendered final PDF matches the manuscript asset pixel for pixel at 110 dpi. |
| Main Table 2, 20D MSE/SE | `results/r2/summaries/revision-mse-summary-20260903T170144JST/mse_table_20d_raw.csv` and `mse_primary_by_seed.csv`; `code/portable/reproduce_tables.py` | All 42 displayed means and 42 SEs match final TeX after rounding. |
| Main 20D calibration figure | `seed_calibration_curves.csv` in `raw-global-null-20260813T162558` and `coverage_by_seed.csv` in the coverage summary; `code/r2/scripts/figures/plot_figure4_compact_preview.R` | 50 seeds per cell; rendered PDF matches the final manuscript asset pixel for pixel at 110 dpi. |
| Main 20D pointwise figure | `results/r2/figure5/figure_20d_localization_r2_preview_summary.csv`; `code/portable/plot_figure5.R` | Rendered PDF matches the manuscript asset pixel for pixel at 110 dpi. The underlying 10-repeat raw RDS set is represented by the included provenance manifest and is not bundled in full. |
| Supplement S1, one-dimensional AdaBoost/GB | 410 saved RDS files in `results/r2/supplement_s1/boosting/`; `code/portable/plot_supplement_s1.R` | Four settings and 50 repeats; rendered PDF matches the manuscript asset pixel for pixel at 110 dpi. |
| Supplement S2/S3, five two-dimensional estimation surfaces | Five selected detailed RDS files under `results/r2/figure_details/2d/`; matching boosting RDS and source metadata; `plot_supplement_eight_panel_release.R` followed by `plot_eight_panel_equal_size_release.R` under `code/r2/scripts/figures/` | All five final PDFs match the manuscript assets pixel for pixel at 110 dpi. |
| Supplement S4, two-dimensional coverage | `coverage_by_seed.csv`, `coverage_by_nominal_mass.csv`, and path-aliased metadata; `code/portable/plot_supplement_s4.R` | 50 seeds per cell; rendered PDF matches the manuscript asset pixel for pixel at 110 dpi. |
| Supplement CDC diagnostic table | `results/r2/tables_2d/submitted/per_seed_metrics.csv` and `stable/per_seed_metrics.csv`; `code/portable/reproduce_tables.py` | Four finite counts, stable means, and SEs match final TeX. |
| Supplement S5, 20D location surface | Seed-021 20D detail RDS and matching boosting RDS; `plot_supplement_eight_panel_release.R` followed by `plot_eight_panel_equal_size_release.R` | Rendered final PDF matches the manuscript asset pixel for pixel at 110 dpi. |
| Supplement S6, transformed 20D generated data | `code/r2/scripts/figures/plot_supplement_s6_compact_preview.R` and 20D model source | Deterministic seed 2; rendered PDF matches the manuscript asset pixel for pixel at 110 dpi. |
| Supplement transformed 20D MSE table | `mse_table_20d_transformed.csv` and `mse_primary_by_seed.csv` in the MSE summary; `code/portable/reproduce_tables.py` | All 42 displayed means and 42 SEs match final TeX after rounding. |
| Supplement one-dimensional unbalanced posterior/coverage | `reference/r1/output/section34_1d/full/`; `code/r2/scripts/figures/plot_figure1_r2_preview.R` with `unbalanced-sample-size` preset | Rendered PDF matches the manuscript asset pixel for pixel at 110 dpi. |

## Seed-matched 2D light settings

| Figure panel | Scenario | n0 | n1 | Data seed |
|---|---|---:|---:|---:|
| Main Figure 3 | local_shift | 5,000 | 5,000 | 1 |
| S2 global shift balanced | global_shift | 5,000 | 5,000 | 1 |
| S2 local dispersion balanced | local_dispersion | 5,000 | 5,000 | 21 |
| S3 global shift unbalanced | global_shift | 9,000 | 1,000 | 1 |
| S3 local shift unbalanced | local_shift | 9,000 | 1,000 | 16 |
| S3 local dispersion unbalanced | local_dispersion | 9,000 | 1,000 | 16 |

Supplement S5 uses the 20D balanced latent-location setting with seed 21. The light command in `README.md` accepts every 2D setting above and the submitted 20D simulation grid. It uses the archived estimator function and the archived light hyperparameters, while selecting one requested setting and seed.
