# Anonymous Round 2 validation status

Checks below were completed on 2026-09-28 with R 4.5.2 on Windows 11 x64.
The active method is ReviewPkg 0.0.0.9001. R1 ReviewPkg 0.0.0.9000 is retained
unchanged for historical use in a separate library.

## Completed on this anonymous candidate

- The R2 anonymous source and the original R2 source were compiled with the same Rtools45 / GCC 14.3.0 toolchain. Twelve reduced 2D/20D cases covering both boosting losses with cross-validation and Bayesian fitting gave exactly identical fitted objects, predictions, posterior draws, and post-fit/post-prediction RNG states.
- Nine core implementation/compiler files are byte-identical to the original R2 source. All 24 historical R1 method files are byte-identical. Across 58 result CSV files, 3,657,717 numeric cells are unchanged.
- Of 992 saved RDS files, 985 are byte-identical. Seven boosting RDS files have only identity metadata adapted; scientific objects and the retained nonidentity metadata compare exactly. Recursive inspection found no remaining targeted direct method/author identifiers, public source commits, or personal paths. Metadata of all 24 included PNGs also passed inspection. This is a targeted identity check, not a guarantee against inference from scientific content.
- The four technical tables were regenerated and all 264 displayed numerical entries matched the local manuscript TeX after rounding to three decimal places.
- All 15 documented technical PDFs were regenerated. At 110 dpi, 14 match the manuscript assets pixel for pixel. Main Figure 2 retains the known panel-geometry difference (17.2554% differing pixels); this difference was present before anonymization.
- The anonymous R2 light run at the displayed 2D local-shift seed 1, with 5,000 observations per group, completed in 355.9 seconds. It produced finite MSE values, a 10,000 by 2 data matrix, finite posterior means, and valid coverage values. Its reduced settings are labeled SMOKE_NOT_FOR_PAPER and do not reproduce the full paper estimates.
- Reduced coverage and boosting workflows completed; the transformed 20D global/null workflow completed two reduced computation checks. Source validation accepts intact files and rejects a modified source before execution. The light runner rejects the R1 library when R2 is requested.
- R sources parse successfully. The release manifest and every ZIP member are SHA-256 verified when sealing the archive; run `python code/verify_bundle.py` after extraction to check the supplied files and seed-level invariants.

Current compact records are in `verification/anonymous_20260928/`. Other files
under `verification/` are retained historical records from the earlier candidate;
its light-run records concern R1 and must not be attributed to the R2 anonymous run.
Detailed author-side logs, build libraries, generated outputs, and the mapping
to original identities remain outside the submission archive.

## Remaining work before final submission

The Section 5 coauthor must confirm final application code, input lineage,
redistribution rights, dependencies, figure commands, and runtime. See
`COAUTHOR_SECTION5.md` and the bracketed fields in the ACC draft. The Main Figure 2
layout difference also remains open. Local checks used a new method library with
existing dependency libraries; a fully isolated dependency installation, a locked
environment, and full peak-resource measurements have not been completed.
The full 50-seed simulations were not rerun as part of anonymization.

This archive is an anonymous author-review candidate. Its adjacent SHA-256 record
identifies this version; integration of coauthor material requires a new manifest
and archive hash before submission.
