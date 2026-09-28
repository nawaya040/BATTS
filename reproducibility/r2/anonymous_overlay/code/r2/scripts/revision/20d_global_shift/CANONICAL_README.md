# 20D global-shift and null canonical run

This simulation evaluates four cells:

- matched global shift, `(n0, n1) = (5000, 5000)`;
- matched global shift, `(n0, n1) = (9000, 1000)`;
- null, `(n0, n1) = (5000, 5000)`;
- null, `(n0, n1) = (9000, 1000)`.

Each cell uses seeds 1--50. BART uses 200 trees, 2,000 burn-in sweeps,
1,000 retained sweeps, and `lambda_0 = 5`. The installed ReviewPkg package must
report Review-Source-ID `466441ba027c17f595f62848082e67c2eb3f2f4f`.

The full `10000 x 1000` posterior log-density-ratio matrix is saved for every
replicate. Each result also has a `.done.rds` sidecar containing its SHA-256,
metrics, calibration curve, and validation summary. A task is skipped on
resume only when both files exist and the checksum and run contract match.

The runner requires exactly four PSOCK workers. Each worker is restricted to
one OpenMP/BLAS thread. The four-cell seed-1 preflight and the full run share
one immutable contract and output directory, so verified preflight results are
reused by the canonical run.
