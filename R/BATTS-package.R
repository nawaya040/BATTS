#' Boosting and Bayesian Additive Trees for Two-Sample Comparison
#'
#' Estimate the square root of a density ratio using boosting or generalized
#' Bayesian additive trees. See [boots()], [batts()], and
#' [eval_balance_weight()] for the public interface.
#'
#' @references Awaya, N., Xu, Y., and Ma, L. (2025). Two-sample comparison
#'   through additive tree models for density ratios.
#'   <https://arxiv.org/abs/2508.03059>.
#' @useDynLib BATTS, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @md
"_PACKAGE"
