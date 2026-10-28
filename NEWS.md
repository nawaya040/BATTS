# BATTS 0.4.0

First CRAN release. This version adds a new interface and is otherwise
backwards compatible: code written for 0.3.0 keeps working and gives the same
fits, predictions and random number use. The algorithms, defaults and C++
code are those of 0.3.0.

## New interface

- New fitting functions `batts_boost()` and `batts_bart()` with settings
  collected in `boost_control()` and `bart_control()`. They give the same
  results as `boots()` and `batts()` with the corresponding settings; the
  fitted object also records the call and the settings.
- `bart_control()` omits the boosting-only arguments of `batts()` (`n_bins`,
  `learn_rate`, `use_gradient`, `n_min_obs_per_node`), which have no effect on
  BART fits. `keep_draws` (saving the posterior forests) stays `FALSE` by default
  because the forests can take over 100 MB with the default settings.
- `batts_boost()` and `batts_bart()` accept a numeric matrix, a data frame with
  numeric columns or, for one variable, a numeric vector as `x`. Group labels
  may be numeric, logical, character or a factor with two values; the first
  level (for non-factors, the smaller value in sort order) is the numerator
  group `p`. The labels are recorded in `fit$group_levels` and shown by
  `print()`.
- Two samples can be passed directly: `batts_boost(x, y = y)` estimates the
  ratio of the density of `x` to that of `y`.

## Fitted objects and methods

- Fits are objects of class `batts_boost` or `batts_bart`, both inheriting
  `batts_fit`; this includes fits from `boots()` and `batts()`. The contents
  of the returned list are unchanged, so code that reads fields such as
  `tree_list` keeps working.
- Added `print()`, `summary()` and `nobs()` methods. Printing a fit now shows a
  short description instead of dumping the whole list; use `unclass(fit)` or
  `str(fit)` for the previous output.
- Added `predict()` and `fitted()` methods and a `density_ratio()` extractor.
  The method is chosen by the fit's class, so there is no `is_Bayes` argument.
  `type` selects what is returned: `"log_ratio"` (the default) gives the
  log-density ratio `2 * log(w) = log(p/q)`, `"ratio"` gives `w ^ 2 = p/q`, and
  `"weight"` gives the balancing weight `w = sqrt(p/q)` that
  `eval_balance_weight()` returns. Group 0 (the first group) is in the
  numerator throughout.
- For a `batts_bart` fit, `summary` and `interval` collapse the posterior
  draws. Each draw is transformed as `type` asks before being summarized, so
  the result summarizes the posterior of the transformed quantity.
- `predict()` accepts a data frame or a vector as `newdata` and matches columns
  by name when the training data had column names. Points outside the fitted
  domain can give `NA` (`outside = "na"`) or be moved to the boundary
  (`outside = "clamp"`); the default (`"error"`) stops and names the offending
  values.
- `predict()` on a BART fit without saved forests explains how to refit with
  `keep_draws = TRUE`.
- Added `plot()` methods, drawn with base graphics. A boosting fit shows the
  cross-validation loss with the selected number of trees, or the fitted
  log-density ratio for one variable; a Bayesian additive trees fit shows the
  trace of the loss temperature, or the posterior mean of the log-density
  ratio with a pointwise credible band for one variable.

## Deprecated

- `boots()`, `batts()` and `eval_balance_weight()` are deprecated in favour of
  `batts_boost()`, `batts_bart()` and `predict()`. They still work, keep their
  arguments and defaults, and return the same results, but warn once per
  session. `?"BATTS-deprecated"` maps the old arguments to the new settings.
  Note that `predict()` returns the log-density ratio by default, whereas
  `eval_balance_weight()` returns the balancing weight; pass
  `type = "weight"` for the older scale.
- The help pages of the deprecated functions are kept, so `?boots` still
  works, but they are marked internal and left out of the reference manual
  and the help index.

## Documentation

- New vignette "Introduction to BATTS" (`vignette("BATTS-introduction")`):
  one-dimensional location and scale examples with credible bands, and the
  two-dimensional global and local shift examples based on the paper.

## Package metadata

- `Description` and `citation("BATTS")` cite the paper,
  arXiv:2508.03059 (doi:10.48550/arXiv.2508.03059).
- `RcppArmadillo` is only linked to, not imported; `graphics` is imported.

# BATTS 0.3.0

- The defaults of `boots()` and `batts()` are now the settings of the paper's
  simulations. `boots()`: gradient boosting (`use_gradient = TRUE`), at most
  1000 trees (`num_trees_max`) selected by 5-fold CV (`K_CV = 5`) and per-tree
  subsampling of one half (`subsample_fraction = 0.5`); depth 4, learning rate
  0.01, 32 bins and margin 0.1 are unchanged. `batts()`: 200 trees
  (`num_trees`), 2000 burn-in iterations (`size_burnin`) and 1000 saved draws
  (`size_backfitting`, previously required); the other defaults are unchanged.
- Calls that relied on the previous defaults now give different results and
  use the random number generator for CV folds and subsamples. Supply the
  previous values explicitly to keep earlier behaviour. Estimation code is
  unchanged; fits with the same explicit settings are identical to 0.2.2.
- The paper's simulations used 0.2.2 (tag `v0.2.2`) with explicit settings.
- The README example and `example.R` use the new defaults, with a shorter
  MCMC run (500 burn-in iterations and 500 draws) for a quick demonstration.

# BATTS 0.2.2

- Add `subsample_fraction` to `boots()` (default `1`, unchanged behavior and
  RNG use). A value below one builds each tree from a random subsample of
  `ceiling(subsample_fraction * m)` training observations, drawn as in `ada`
  with `bag.frac`: one observation per group, the rest uniformly without
  replacement regardless of group. Splits, gradients and leaf values use the
  subsample; all weights are updated, and the forward-stagewise normalization
  and CV losses use the whole training set and held-out folds, respectively.
- The internal `run_adaboost()` entry point takes the per-tree subsample
  indices as an additional argument; `batts()` does not subsample.

# BATTS 0.2.1

- Reject invalid burn-in and retained-draw counts before fitting (U22).
- Reject move probabilities outside the documented positive, length-three,
  unit-sum contract in R and C++; the sum tolerance is 1e-12 (U23).
- These guards preserve valid-input calculations and RNG consumption. Invalid
  inputs now fail before fitting. Inverse-Gaussian limitation U16 is unchanged.

# BATTS 0.2.0

- Require a positive integer `thin` and validate prediction matrix dimensions
  and finite values before passing data to C++ (U14, U17).
- Breaking API change: remove `update_lambda`, `lambda_prior_parameters`, and
  `alpha_cutpoint`. Leaf scale stays fixed at `num_trees * lambda_0`; cut
  spacing/proposals use the previous default. Remove these named arguments
  from existing calls. Nondefault cut grids and scale updating are no longer
  supported. The C++ interface is regenerated with the reduced argument list.
- Document inverse-Gaussian cancellation under extreme parameters as known
  limitation U16; its numerical formula is unchanged by author decision.

# BATTS 0.1.0

Official distribution: https://github.com/nawaya040/BATTS.

## Fixes integrated from development

This release integrates commits `6ea6722` through `909ea357` after the
previous default-branch commit `6f625ba`. They correct forward-stagewise
normalization and prediction constants, posterior-forest storage and guards,
CV group handling, gradient CV scaling, split loss and split-boundary rules,
PRUNE indices, and empty-leaf sampling and GROW moves. They add input guards
and restrict Bayesian initialization to unsplit trees. They also reduce
unnecessary copies, history allocation and internal-node index storage.

These existing changes can alter estimates and RNG streams. A run using
0.1.0 must record its own version and settings; numerical identity with the
previous public version or every reported paper result is not asserted.

## Additional release fixes

- Rename the prediction-only Node type to PredictionNode, removing its
  one-definition-rule collision with the fitting Node type.
- Require C++17 for compatibility with recent RcppArmadillo.
- Replace placeholder author/maintainer metadata, add API help and regression
  tests, and document the official release installation.

The new type-name fix preserves mathematical operations and RNG calls.
Build-level equivalence and limitations are documented in [inst/VALIDATION.md](inst/VALIDATION.md).
No model, prior, canonical experiment setting or saved result is changed
by this release preparation.
