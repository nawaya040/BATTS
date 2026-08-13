# Revision computation plan

This plan is preparation only. A canonical launch still requires separate
approval and the exact `--confirm-canonical=YES` guard.

## Scientific units

| Phase | Scheduler rows | Scientific units | Contents |
|---|---:|---:|---|
| Transformed BART | 1 | 200 | global shift/null, balanced/unbalanced, 50 seeds |
| Boosting | 400 | 400 | GB, FS, Ada DRT exponential-loss primary and classification diagnostic |
| Kernel | 800 | 800 | KLIEP and uLSIF |
| CDC | 400 | 400 | submitted and numerically stable versions, dependent on Ada DRT |
| Coverage correction | 900 | 900 | 1D 200, 2D 300, 20D raw 200, 20D transformed 200 |
| Total | 2,501 | 2,700 | Existing raw global/null BART results are not rerun |

The new 20D global/null grid uses the same raw generator for every method.
The transformed version applies the coordinatewise map
`qbeta(pnorm(x / reference_sd), 0.5, 10)` without consuming RNG. The analytic
log-density ratio is unchanged by this common invertible transformation.

## Execution order

1. Transformed BART batch.
2. Boosting and kernel phases. These can run independently.
3. CDC after its boosting dependencies have passed checksum verification.
4. Corrected coverage jobs, preferably after comparator phases to limit RAM.

All job-level runners refuse overwrite. Resume mode verifies the stored
SHA-256 and scientific identity before skipping. The phase launcher verifies
the job-grid SHA-256 and writes a status line with an updated remaining-time
estimate every 15 minutes by default.

After all comparator phases, `20d_global_shift/summarize_global_null_comparators.R`
verifies every result against its sidecar, requires matching generated-data
hashes across methods, and writes job-level and aggregated metric tables into
a new summary directory.
