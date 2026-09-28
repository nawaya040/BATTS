# Coverage-only reruns

These scripts recompute Bayesian credible-interval coverage with symmetric
quantile indices. They do not run GB, FS, AdaBoost, KLIEP, uLSIF, or CDC.

The corrected upper index is centralized in `coverage_common.R`:

```r
upper_index <- n_probs - lower_index + 1L
```

## Safety modes

The default mode is `smoke`. Smoke outputs are written under an explicit
`smoke/` directory and carry the manifest status `SMOKE_NOT_FOR_PAPER`.

Canonical mode requires the exact guard:

```text
--mode=canonical --confirm-canonical=YES
```

Existing outputs are never overwritten. With `--resume=true`, an existing
result is skipped only after its SHA-256 hash is verified against the sidecar
manifest.

## Required arguments

Every runner requires:

```text
--seed=<integer>
--reviewpkg-lib=<R library containing ReviewPkg>
--output-dir=<new or existing output root>
```

The 2D and 20D runners also require `--scenario` in canonical mode. The 20D
runner accepts `--transformed=true|false`.

## Examples

Smoke:

```text
Rscript scripts/coverage/run_coverage_2d.R `
  --mode=smoke --seed=1 `
  --reviewpkg-lib=C:/path/to/r-library `
  --output-dir=C:/path/to/new-output-root
```

One canonical job:

```text
Rscript scripts/coverage/run_coverage_2d.R `
  --mode=canonical --confirm-canonical=YES `
  --scenario=local_dispersion --n0=5000 --n1=5000 --seed=1 `
  --reviewpkg-lib=C:/path/to/r-library `
  --output-dir=C:/path/to/new-output-root
```

## Canonical job grid

- 1D: `(500,500)`, `(100,900)`, `(2500,2500)`, `(500,4500)`;
- 2D scenarios: `global_shift`, `local_shift`, `local_dispersion`;
- 2D sample sizes: `(5000,5000)`, `(9000,1000)`;
- 20D scenarios: `latent_location_shift`, `latent_dispersion`;
- 20D sample sizes: `(5000,5000)`, `(9000,1000)`;
- seeds: 1 through 50;
- transformed 20D is a separate optional grid.

The ordinary 1D, 2D, and 20D grids contain 700 jobs. Including transformed
20D gives 900 jobs. Use a controlled parallel schedule for a full run to limit memory use.

## Output

Each job produces:

- `<job-id>.rds`: metadata and the corrected coverage result;
- `<job-id>.manifest.rds`: configuration, source hashes, environment,
  compiler settings, RNG information, CPU/elapsed time, and output SHA-256.

Jobs are sequential internally. Parallelism belongs in the external scheduler,
at the job level.
