#' Deprecated functions in BATTS
#'
#' These functions still work and give the same results as before, but warn
#' once per session. Use the replacements below.
#'
#' | Deprecated | Replacement |
#' |---|---|
#' | [boots()] | [batts_boost()] with [boost_control()] |
#' | [batts()] | [batts_bart()] with [bart_control()] |
#' | [eval_balance_weight()] | [predict()] (`type = "weight"` for the same scale) or [density_ratio()] |
#'
#' Arguments of [boots()] and [batts()] that moved into the control functions:
#'
#' | Old argument | New setting |
#' |---|---|
#' | `data`, `group_labels` | `x`, `group` |
#' | `max_resol` | `depth` (boosting) |
#' | `n_min_obs_per_node` | `min_obs_per_node` (boosting) |
#' | `size_burnin` | `burnin` |
#' | `size_backfitting` | `draws` |
#' | `omega_prior_parameters` | `omega_prior` |
#' | `tree_priors` | `tree_prior` |
#' | `output_BART_ensembles` | `keep_draws` |
#'
#' `n_ratio_per_node` accepted only its default and has no replacement. The
#' [batts()] arguments `max_resol`, `n_bins`, `learn_rate`, `use_gradient`
#' and `n_min_obs_per_node` have no effect on a BART fit and have no
#' replacement in [bart_control()].
#'
#' `eval_balance_weight(fit, points)` returns the balancing weight
#' `w = sqrt(p/q)`; `predict(fit, points)` returns the log-density ratio
#' `2 * log(w)` by default.
#' @name BATTS-deprecated
#' @keywords internal
#' @md
NULL

.batts_warned = new.env(parent = emptyenv())

# Warn once per session that old() is deprecated in favour of new().
.batts_deprecated = function(old, new){
  if(isTRUE(.batts_warned[[old]])) return(invisible(NULL))
  assign(old, TRUE, envir = .batts_warned)
  warning(warningCondition(
    sprintf("%s() is deprecated; use %s() instead (see ?`BATTS-deprecated`).\n%s() still works and gives the same result. This warning is shown once per session.",
            old, new, old),
    class = "deprecatedWarning"))
}
