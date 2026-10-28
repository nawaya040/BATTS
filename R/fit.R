.check_count = function(v, name, min){
  if(!is.numeric(v) || length(v) != 1L || !is.null(dim(v)) || !is.finite(v) ||
     v != floor(v) || v < min || v > .Machine$integer.max){
    stop(sprintf("%s must be a single integer of at least %d", name, min), call. = FALSE)
  }
}

.check_positive = function(v, name, len = 1L){
  if(!is.numeric(v) || length(v) != len || !is.null(dim(v)) ||
     any(!is.finite(v)) || any(v <= 0)){
    stop(sprintf("%s must be %s", name,
                 if(len == 1L) "a single positive number"
                 else sprintf("%d positive numbers", len)), call. = FALSE)
  }
}

.check_number = function(v, name){
  if(!is.numeric(v) || length(v) != 1L || !is.null(dim(v)) || !is.finite(v)){
    stop(sprintf("%s must be a single finite number", name), call. = FALSE)
  }
}

.check_flag = function(v, name){
  if(!is.logical(v) || length(v) != 1L || is.na(v)){
    stop(sprintf("%s must be TRUE or FALSE", name), call. = FALSE)
  }
}

#' Settings for boosting fits
#'
#' Collect the settings of [batts_boost()]. The defaults are the settings of
#' the paper's simulations and equal the defaults of [boots()].
#'
#' @param num_trees_max Maximum number of boosting trees.
#' @param K_CV Number of group-stratified cross-validation folds used to select
#'   the number of trees, or zero to fit `num_trees_max` trees without
#'   cross-validation.
#' @param depth Maximum depth of each tree.
#' @param learn_rate Shrinkage factor of each boosting step.
#' @param n_bins Number of bins used to construct the candidate cut points.
#' @param margin_scale Fraction of each training range added at both ends of
#'   the fitted domain. A negative value uses the unit box and requires the
#'   data to lie in it.
#' @param subsample_fraction Fraction of the training observations used to
#'   construct each tree, in (0, 1].
#' @param min_obs_per_node Minimum count from each group in a child node.
#' @param use_gradient Logical; gradient boosting when `TRUE`, forward-stagewise
#'   boosting when `FALSE`.
#' @return An object of class `boost_control`.
#' @seealso [batts_boost()], [boots()] for the meaning of each setting.
#' @examples
#' boost_control(depth = 2)
#' @md
#' @export
boost_control = function(num_trees_max = 1000, K_CV = 5, depth = 4,
                         learn_rate = 0.01, n_bins = 32, margin_scale = 0.1,
                         subsample_fraction = 0.5, min_obs_per_node = 1,
                         use_gradient = TRUE){
  .check_count(num_trees_max, "num_trees_max", 1)
  .check_count(depth, "depth", 1)
  .check_positive(learn_rate, "learn_rate")
  .check_count(n_bins, "n_bins", 2)
  .check_number(margin_scale, "margin_scale")
  .check_flag(use_gradient, "use_gradient")
  out = list(num_trees_max = num_trees_max, K_CV = K_CV, depth = depth,
             learn_rate = learn_rate, n_bins = n_bins, margin_scale = margin_scale,
             subsample_fraction = subsample_fraction,
             min_obs_per_node = min_obs_per_node, use_gradient = use_gradient)
  class(out) = "boost_control"
  out
}

#' Settings for Bayesian additive tree fits
#'
#' Collect the settings of [batts_bart()]. The defaults are the settings of
#' the paper's simulations and equal the defaults of [batts()]. The
#' boosting-only arguments of [batts()] (`n_bins`, `learn_rate`,
#' `use_gradient` and `n_min_obs_per_node`) are omitted because they do not
#' affect a BART fit.
#'
#' @param num_trees Number of trees in the ensemble.
#' @param burnin Number of discarded backfitting iterations. `NULL` uses
#'   `floor(draws / 2)`.
#' @param draws Number of saved posterior draws.
#' @param thin Number of backfitting sweeps per iteration.
#' @param prob_moves Probabilities of the GROW, PRUNE and CHANGE moves.
#' @param lambda_0 Leaf-prior scale per tree.
#' @param omega_prior Shape and rate of the Gamma prior on the loss
#'   temperature.
#' @param tree_prior Two tree-depth prior parameters: the split probability at
#'   depth d is `tree_prior[1] / (1 + d)^tree_prior[2]`.
#' @param margin_scale Fraction of each training range added at both ends of
#'   the fitted domain.
#' @param keep_draws Logical; save the posterior forests so that the fit can be
#'   evaluated at new points with [predict()]. Saving them does not change the
#'   draws or the use of random numbers, but can make the fitted object large
#'   (over 100 MB with the default numbers of trees and draws).
#' @return An object of class `bart_control`.
#' @seealso [batts_bart()], [batts()] for the meaning of each setting.
#' @examples
#' bart_control(num_trees = 100, burnin = 300, draws = 301, keep_draws = TRUE)
#' @md
#' @export
bart_control = function(num_trees = 200, burnin = 2000, draws = 1000, thin = 1,
                        prob_moves = c(1/3, 1/3, 1/3), lambda_0 = 5,
                        omega_prior = c(1, 1), tree_prior = c(0.95, 2.0),
                        margin_scale = 0.1, keep_draws = FALSE){
  .check_count(num_trees, "num_trees", 1)
  .check_positive(lambda_0, "lambda_0")
  .check_positive(omega_prior, "omega_prior", 2L)
  .check_positive(tree_prior, "tree_prior", 2L)
  .check_number(margin_scale, "margin_scale")
  .check_flag(keep_draws, "keep_draws")
  out = list(num_trees = num_trees, burnin = burnin, draws = draws, thin = thin,
             prob_moves = prob_moves, lambda_0 = lambda_0,
             omega_prior = omega_prior, tree_prior = tree_prior,
             margin_scale = margin_scale, keep_draws = keep_draws)
  class(out) = "bart_control"
  out
}

#' Fit a two-sample density ratio by boosting
#'
#' Fit the balancing weight `w = sqrt(p/q)` by tree boosting, where `p` is the
#' density of the first group and `q` the density of the second. This calls
#' [boots()] with the settings in `control` and gives the identical result.
#'
#' @param x Observations in rows and variables in columns: a numeric matrix, a
#'   data frame with numeric columns, or a numeric vector for one variable.
#'   Values must be finite.
#' @param group Two-valued labels, one per row of `x`: numeric, logical,
#'   character or factor. The first group (the first factor level, otherwise the
#'   smaller value in sort order; 0 for 0/1 labels) has density `p`, the
#'   numerator.
#' @param y Instead of `group`, a second sample with the same columns as `x`.
#'   Then `x` has density `p` and `y` has density `q`.
#' @param control Settings created by [boost_control()].
#' @param quiet Logical; suppress progress messages.
#' @return An object of class `batts_boost`, inheriting `batts_fit`, as
#'   returned by [boots()], with the call, the settings, the group labels
#'   (`group_levels`, numerator first) and the column names
#'   (`variable_names`) added. Use [predict()], [fitted()] and
#'   [density_ratio()] to evaluate it.
#' @details Supply exactly one of `group` and `y`. Set the seed before
#'   fitting; cross-validation folds and subsamples use the R random number
#'   generator.
#' @seealso [boost_control()], [batts_bart()], [batts-predict]
#' @examples
#' set.seed(1)
#' p_sample <- rnorm(50)
#' q_sample <- rnorm(50, 0.5)
#' fit <- batts_boost(p_sample, y = q_sample,
#'                    control = boost_control(num_trees_max = 50, depth = 2),
#'                    quiet = TRUE)
#' fit
#' predict(fit, c(-1, 0, 1))
#'
#' # The same data as one sample with group labels; the first level is p
#' x <- c(p_sample, q_sample)
#' g <- factor(rep(c("p", "q"), each = 50))
#' ctl <- boost_control(num_trees_max = 50, depth = 2)
#' set.seed(2); a <- batts_boost(x, g, control = ctl, quiet = TRUE)
#' set.seed(2); b <- batts_boost(p_sample, y = q_sample, control = ctl, quiet = TRUE)
#' identical(predict(a, 0), predict(b, 0))
#' @md
#' @export
batts_boost = function(x, group = NULL, y = NULL, control = boost_control(),
                       quiet = FALSE){
  if(!inherits(control, "boost_control")){
    stop("control must be created by boost_control()", call. = FALSE)
  }
  .check_flag(quiet, "quiet")
  input = .fit_input(x, group, y)
  .check_constant_columns(input$x, control$margin_scale)
  out = boots(input$x, input$group,
              num_trees_max = control$num_trees_max,
              K_CV = control$K_CV,
              max_resol = control$depth,
              learn_rate = control$learn_rate,
              n_bins = control$n_bins,
              n_min_obs_per_node = control$min_obs_per_node,
              margin_scale = control$margin_scale,
              use_gradient = control$use_gradient,
              quiet = quiet,
              subsample_fraction = control$subsample_fraction)
  out$call = match.call()
  out$control = control
  out$group_levels = input$levels
  out$variable_names = input$variable_names
  out
}

#' Fit a two-sample density ratio by Bayesian additive trees
#'
#' Draw the balancing weight `w = sqrt(p/q)` from its generalized posterior,
#' where `p` is the density of the first group and `q` the density of the
#' second. This calls [batts()] with the settings in `control` and gives the
#' identical result.
#'
#' @inheritParams batts_boost
#' @param control Settings created by [bart_control()].
#' @return An object of class `batts_bart`, inheriting `batts_fit`, as
#'   returned by [batts()], with the call, the settings, the group labels
#'   (`group_levels`, numerator first) and the column names
#'   (`variable_names`) added. Use [predict()], [fitted()] and
#'   [density_ratio()] to evaluate it; evaluating it at new points requires
#'   `bart_control(keep_draws = TRUE)`.
#' @details Supply exactly one of `group` and `y`. Set the seed before
#'   fitting. A short run does not establish convergence; the returned
#'   intervals are pointwise.
#' @seealso [bart_control()], [batts_boost()], [batts-predict]
#' @examples
#' set.seed(2)
#' # Small demonstration; these settings are not a convergence recommendation.
#' fit <- batts_bart(rnorm(30), y = rnorm(30, 0.5),
#'                   control = bart_control(num_trees = 10, burnin = 20,
#'                                          draws = 21, keep_draws = TRUE),
#'                   quiet = TRUE)
#' fit
#' predict(fit, c(-1, 0, 1), interval = TRUE)
#' @md
#' @export
batts_bart = function(x, group = NULL, y = NULL, control = bart_control(),
                      quiet = FALSE){
  if(!inherits(control, "bart_control")){
    stop("control must be created by bart_control()", call. = FALSE)
  }
  .check_flag(quiet, "quiet")
  input = .fit_input(x, group, y)
  .check_constant_columns(input$x, control$margin_scale)
  out = batts(input$x, input$group,
              num_trees = control$num_trees,
              margin_scale = control$margin_scale,
              size_burnin = control$burnin,
              size_backfitting = control$draws,
              thin = control$thin,
              prob_moves = control$prob_moves,
              lambda_0 = control$lambda_0,
              omega_prior_parameters = control$omega_prior,
              tree_priors = control$tree_prior,
              output_BART_ensembles = control$keep_draws,
              quiet = quiet)
  out$call = match.call()
  out$control = control
  out$group_levels = input$levels
  out$variable_names = input$variable_names
  out
}
