#' Boosting and Bayesian Additive Trees for Two-Sample Comparison
#'
#' Estimate the density ratio of two samples using boosting or generalized
#' Bayesian additive trees.
#'
#' - [batts_boost()] fits by boosting, with settings in [boost_control()].
#' - [batts_bart()] draws from the generalized posterior, with settings in
#'   [bart_control()].
#' - [predict()], [fitted()] and [density_ratio()] evaluate a fit; see
#'   [batts-predict]. [print()], [summary()] and [plot()][batts-plot] describe
#'   it.
#'
#' The earlier interface, [boots()], [batts()] and [eval_balance_weight()],
#' is deprecated but still works; see [BATTS-deprecated].
#'
#' @references Awaya, N., Xu, Y., and Ma, L. (2025). Two-sample comparison
#'   through additive tree models for density ratios. arXiv:2508.03059.
#'   \doi{10.48550/arXiv.2508.03059}.
#' @useDynLib BATTS, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @md
"_PACKAGE"
