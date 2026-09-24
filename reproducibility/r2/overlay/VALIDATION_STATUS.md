# Validation status of this review candidate

## Completed

- The current source manifest lists more than 1,250 files. Every original copy was SHA-256 checked, and `python code/verify_bundle.py` passes for all listed release files. Thirteen provenance text/CSV files had local absolute paths replaced by stable aliases; the original and release hashes are recorded separately. Numerical summary columns and RDS objects were left intact.
- The included BATTS source was extracted from the specified Git commit. The Round 2 workflow source commit was identified separately.
- The seed-level checks found 112 complete MSE settings (5,600 rows) and 72 complete coverage settings (356,400 rows), each with seeds 1–50. Six 2D detail RDS files, one 20D detail RDS file, and all selected boosting RDS files matched their checksum records.
- The technical Table 1, Table 2, transformed 20D table, and CDC diagnostic table were regenerated from bundled seed-level CSV files. All 264 displayed numeric entries matched the final local TeX after rounding. The generated CSVs and comparison result are in `verification/tables/`. The command is `python code/portable/reproduce_tables.py <new-output-dir>`; `code/portable/compare_tables_to_tex.py` performs the comparison when the manuscript TeX files are supplied.
- The main and supplementary figure drawing commands generated PDFs in isolated directories. At 110 dpi, 14 regenerated PDFs matched the corresponding manuscript PDFs pixel for pixel: both 1D Figure 1 variants, Main Figure 3, the main 20D calibration and pointwise figures, S1, five S2/S3 surfaces, S4, S5, and S6. The per-figure comparison CSVs are in `verification/`. PDF byte hashes can differ because of PDF metadata.
- The 2D light run for the Main Figure 3 setting completed at seed 1 and produced valid summary and detailed RDS files. The 20D balanced latent-location light run completed at seed 21, with a 10,000-observation detail object and finite MSE values. Both used the archived Round 1 light parameters and are labeled `SMOKE_NOT_FOR_PAPER`. On this PC they took approximately 5 and 6 minutes, respectively. Output SHA-256 values are in `verification/light_run_checksums.csv`; the generated RDS files are retained in the adjacent `validation_outputs/` directory for author inspection.

## Outstanding before submission

- Main Figure 2 renders from the included fixed-seed model code, but the panel geometry differs from the final PDF. At 110 dpi, 17.3% of pixels differ. The plotted scenarios appear visually consistent; the exact reason for the layout difference has not been established.
- A clean-environment light run and a measured peak-resource profile remain outstanding. The completed local light outputs are demonstrations and can differ from full results.
- Pin or otherwise document exact dependency versions, complete a clean-environment installation check, and record a final run manifest.
- Obtain the Section 5 coauthor's final code, data lineage, permissions, and figure confirmation.
- Review the final package size and the journal upload limit, then create and inspect the single submission ZIP.

These checks used the existing local R 4.5.2 installation and already installed packages. The drawing tests and the 2D/20D light runs establish execution on this machine; a clean-environment test and any full canonical rerun remain separate tasks.
