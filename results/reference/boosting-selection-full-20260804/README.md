# Canonical boosting-selection results

This directory records the compact, auditable reference summary for canonical
run `boosting-selection-full-20260804`. The raw RDS outputs are intentionally
not tracked in Git.

## Run scope and provenance

- Source commit: `66c74aaa838f07a04141a58ef611c1f5ba0d68ed`
- Design: 14 cells, 50 seeds per cell, 700 jobs total
- CV folds: 5
- Maximum trees: 1,000
- Learning rate: 0.01
- AdaBoost: depth 4, bag fraction 0.5
- Proposed GB and FS: depth 4
- Evaluation: symmetric training-sample estimation MSE against the known
  simulation truth, `(mse_group0 + mse_group1) / 2`

`run_provenance.csv` records the R version, RNG kind, package versions, source
hashes, and installed BATTS hashes. `output_checksums.csv` records the SHA-256
hash and byte size of every external result file.

## Completion and audit

The run finished on 2026-08-05 at 23:49 local time. All 700 result/manifest
pairs passed the final audit:

- 700 unique job IDs and 700 unique output hashes;
- manifest hashes and file sizes matched the result files;
- source commit, source hashes, package hashes, canonical settings, and seed
  mappings were consistent across the run;
- all saved fold curves had the expected `5 x 1000` dimensions and finite
  values;
- saved tree selections agreed with the corresponding CV argmins;
- all estimates and MSE values were finite, and symmetric MSE agreed with the
  mean of the two group-specific MSE values;
- all 700 logs contained a completion marker, with no error or warning pattern.

The launcher wrote `FAILED` for completed jobs and consequently created
`GRID_HAS_FAILURES.txt`. This was traced to its handling of a blank/null wrapper
exit-code property. It is a bookkeeping defect, not evidence of numerical job
failure. The 700 result/manifest pairs and completion logs are the authoritative
completion evidence.

## Main descriptive result

The proposed method had the lowest mean MSE in 10 of 14 design cells:

- all seven unbalanced cells;
- all three balanced 2D cells.

Exponential-loss CV AdaBoost had the lowest mean MSE in all four balanced 20D
cells. Thus the result supports a conditional advantage under imbalance, not an
unconditional dominance claim.

| Family | Scenario | Sampling | Transformed | Ada Exp MSE | GB MSE | FS MSE | GB vs Ada | FS vs Ada |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2D | Global shift | Balanced | No | 0.042687 | 0.032821 | 0.035590 | +23.1% | +16.6% |
| 2D | Global shift | Unbalanced | No | 0.107267 | 0.063131 | 0.072063 | +41.1% | +32.8% |
| 2D | Local shift | Balanced | No | 0.047474 | 0.034850 | 0.035251 | +26.6% | +25.7% |
| 2D | Local shift | Unbalanced | No | 0.118450 | 0.067472 | 0.069804 | +43.0% | +41.1% |
| 2D | Local dispersion | Balanced | No | 0.120912 | 0.108047 | 0.111468 | +10.6% | +7.8% |
| 2D | Local dispersion | Unbalanced | No | 0.179518 | 0.133558 | 0.131926 | +25.6% | +26.5% |
| 20D | Latent location shift | Balanced | No | 0.069546 | 0.073125 | 0.075718 | -5.1% | -8.9% |
| 20D | Latent location shift | Balanced | Yes | 0.069516 | 0.076315 | 0.079249 | -9.8% | -14.0% |
| 20D | Latent location shift | Unbalanced | No | 0.173123 | 0.125492 | 0.136019 | +27.5% | +21.4% |
| 20D | Latent location shift | Unbalanced | Yes | 0.173328 | 0.128805 | 0.139849 | +25.7% | +19.3% |
| 20D | Latent dispersion | Balanced | No | 0.136964 | 0.150521 | 0.157933 | -9.9% | -15.3% |
| 20D | Latent dispersion | Balanced | Yes | 0.137156 | 0.162648 | 0.170871 | -18.6% | -24.6% |
| 20D | Latent dispersion | Unbalanced | No | 0.263679 | 0.227554 | 0.238258 | +13.7% | +9.6% |
| 20D | Latent dispersion | Unbalanced | Yes | 0.264388 | 0.235828 | 0.246965 | +10.8% | +6.6% |

Positive percentages denote lower MSE than exponential-loss CV AdaBoost.
`cell_summary.csv` contains Monte Carlo standard errors, selected-tree summaries,
and per-replicate win rates.

Classification-error CV selected substantially fewer trees under imbalance
and produced much larger density-ratio estimation MSE. Exponential-loss CV is
therefore the primary AdaBoost comparator. Exponential-loss and balancing-loss
CV selected identical tree counts and produced identical MSE in all 700 jobs;
their saved loss curves differ only by a constant.

## Maximum-tree boundary

The primary exponential-loss AdaBoost selection reached 1,000 trees in 3 of
700 jobs (0.43%). The balancing-loss diagnostic duplicated those same three
selections. All three occurred in the 20D latent-dispersion, balanced,
transformed cell, at seeds 30, 40, and 43.

The legacy classification-error criterion reached 1,000 trees in one additional
job: 20D latent-location shift, balanced, untransformed, seed 47. Proposed GB
and FS had no 1,000-tree boundary hits.

Because primary boundary hits were rare, a manuscript note and sensitivity
qualification are appropriate. The affected transformed balanced-dispersion
cell should not be used to claim that 1,000 trees was universally sufficient.
Exact records are in `upper_bound_hits.csv`.

## Files

- `cell_summary.csv`: one row per design cell, including tree counts, MSE,
  Monte Carlo standard errors, relative improvements, win rates, and boundary
  counts;
- `upper_bound_hits.csv`: exact job and criterion for every 1,000-tree hit;
- `output_checksums.csv`: SHA-256 and byte size for all 700 external RDS files;
- `run_provenance.csv`: compact run configuration and environment provenance.
