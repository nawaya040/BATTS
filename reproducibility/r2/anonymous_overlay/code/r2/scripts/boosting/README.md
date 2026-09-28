# Boosting tree-selection diagnostics

This directory contains a separate, auditable workflow for comparing tree
selection in the submitted Real AdaBoost comparator and the proposed boosting
methods. It does not alter the submitted results or the coverage-only reruns.

## Interpretation fixed before canonical runs

- `classification_error` is the legacy sensitivity analysis. Its final tree
  count applies the submitted minimum of 10 trees.
- `exponential_loss` is the primary Real AdaBoost criterion.
- `balancing_loss` is a counterfactual diagnostic. It is not presented as a
  natural AdaBoost objective or as a co-primary tuning rule.

All three AdaBoost criteria are evaluated on the same stratified folds and the
same fitted tree path. The balancing criterion is evaluated from the
out-of-fold density-ratio estimate implied by the AdaBoost margin,

```text
log r_t(x) = -2 F_t(x) + log(n1_train / n0_train),
L_bal(t) = mean_group0(exp(-log r_t / 2))
         + mean_group1(exp( log r_t / 2)).
```

The implementation stores the log loss for numerical stability. Fold log
losses are averaged exactly as in the preceding exponential-loss audit.

For the proposed GB and FS methods, the workflow records the selected number
of trees, fold-level and aggregate CV curves, fold-level argmins, upper-bound
hits, and node counts without changing `ReviewPkg::fit_boosting_model()`.

Performance metrics for AdaBoost, GB, and FS are evaluated on the training
observations, matching the submitted simulation design. Independent test
evaluation is intentionally excluded because `ReviewPkg::evaluate_density_ratio()`
does not extrapolate beyond the fitted sample space.

## Safety modes

The default mode is `smoke`. Smoke runs use two folds, at most 60 trees, small
samples, and outputs labelled `SMOKE_NOT_FOR_PAPER`.

Canonical mode requires the exact guard:

```text
--mode=canonical --confirm-canonical=YES
```

Canonical settings are fixed at five folds, 1,000 maximum boosting trees,
learning rate 0.01, AdaBoost depth 4, and bag fraction 0.5. Existing outputs
are never overwritten. `--resume=true` skips an existing result only after
its identity and SHA-256 hash are verified.

## Example smoke run

```text
Rscript scripts/boosting/run_boosting_selection.R `
  --mode=smoke --family=2d --scenario=local_shift --seed=1 `
  --r-lib=C:/path/to/isolated-r-library `
  --output-dir=C:/path/to/new-output-root
```

## Canonical grid

- 2D scenarios: `global_shift`, `local_shift`, `local_dispersion`;
- ordinary 20D scenarios: `latent_location_shift`, `latent_dispersion`;
- transformed 20D repeats the two 20D scenarios with `--transformed=true`;
- sample sizes: `(5000,5000)` and `(9000,1000)`;
- seeds: 1 through 50.

This gives 14 settings and 700 setting-seed jobs. Allocate sufficient runtime
and memory before starting the full grid.
