# 20D matched global-shift pilot

Status: reviewer-driven diagnostic workflow. It does not modify the legacy
experiment drivers or official results.

## Scientific design

For each group, four-dimensional latent observations are generated as

```text
Z0 ~ N(-0.5 e1, I4)
Z1 ~ N( 0.5 e1, I4)
```

and projected into 20 dimensions using the same randomly generated orthonormal
loading structure as the existing 20D experiments:

```text
X = U Z + epsilon,  epsilon ~ N(0, 0.1^2 I20).
```

There is no shared diffuse mixture component and no marginal transformation.
The true log-density ratio is the global linear function

```text
log(p0(x) / p1(x)) = -(U e1)' x / 1.01.
```

The pilot evaluates `(n0, n1) = (5000, 5000)` and `(9000, 1000)` for seeds
1--10. The installed ReviewPkg package must report Review-Source-ID
`466441ba027c17f595f62848082e67c2eb3f2f4f`.

## Execution policy

- Outputs require an explicit path outside official results.
- Existing output files are never overwritten.
- Pilot execution uses exactly two independent PSOCK workers.
- Each worker is restricted to one OpenMP/BLAS thread.
- Each seed is saved atomically as a separate RDS file.
- The full posterior draw matrix is not retained.

## Commands

Run a reduced smoke test:

```powershell
Rscript --vanilla run_20d_global_shift_pilot.R `
  --mode=smoke `
  --workers=1 `
  --output-dir=<isolated-smoke-directory>
```

Run one full replicate for each sample-size cell:

```powershell
Rscript --vanilla run_20d_global_shift_pilot.R `
  --mode=preflight `
  --workers=2 `
  --output-dir=<isolated-preflight-directory>
```

Run the 10-replicate pilot:

```powershell
Rscript --vanilla run_20d_global_shift_pilot.R `
  --mode=pilot `
  --workers=2 `
  --output-dir=<isolated-pilot-directory>
```

Summarize the 20 pilot results:

```powershell
Rscript --vanilla summarize_20d_global_shift_pilot.R `
  --input-dir=<isolated-pilot-directory> `
  --output-dir=<new-summary-directory> `
  --expected=20
```

Run the unbalanced calibration simulation:

```powershell
Rscript --vanilla run_20d_global_shift_pilot.R `
  --mode=calibration-unbalanced `
  --workers=2 `
  --output-dir=<isolated-calibration-run-directory>
```

Create the provisional calibration plot and tables:

```powershell
Rscript --vanilla plot_unbalanced_calibration_curve.R `
  --input-dir=<isolated-calibration-run-directory> `
  --output-dir=<new-calibration-summary-directory>
```
