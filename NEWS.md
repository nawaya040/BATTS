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
