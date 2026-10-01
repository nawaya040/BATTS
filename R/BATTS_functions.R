.validate_group_labels = function(data, group_labels){
  if(is.null(nrow(data)) || !is.numeric(group_labels) ||
     !is.null(dim(group_labels)) ||
     length(group_labels) != nrow(data) ||
     anyNA(group_labels) || !all(is.finite(group_labels)) ||
     !all(group_labels %in% c(0, 1)) ||
     !any(group_labels == 0) || !any(group_labels == 1)){
    stop("group_labels must have one 0 or 1 per data row, with both groups present")
  }
}

#' Boosting for a two-sample density ratio
#'
#' Fit an additive tree estimate of the balancing weight w = sqrt(p/q).
#' The full log-density ratio is obtained as `2 * log(w)`.
#'
#' @param data Numeric matrix with observations in rows and variables in columns.
#'   Values must be finite. With nonnegative `margin_scale`, each column must vary.
#' @param group_labels Numeric vector of zeros and ones, one per row of `data`,
#'   with both groups present. Group 0 has density p and group 1 has density q.
#' @param max_resol Maximum boosting tree depth. For `batts`, must be zero
#'   so that all initial trees are unsplit.
#' @param learn_rate Boosting shrinkage factor; use a positive value.
#' @param n_bins Number of bins used to construct the boosting cut grid.
#'   For root-initialized `batts`, retained for interface compatibility.
#' @param n_min_obs_per_node Minimum count from each group in a boosting child.
#' @param n_ratio_per_node Reserved argument. Only the default `1e-100` is supported.
#' @param margin_scale Nonnegative fraction of each training range added at both
#'   ends of the fitted domain. A negative value uses the fixed unit box and
#'   requires that the supplied data already lie in that box.
#' @param use_gradient Logical; use gradient boosting when TRUE and
#'   forward-stagewise boosting when FALSE. Initialization in `batts` is unsplit.
#' @param quiet Logical; suppress progress messages.
#' @param num_trees_max Maximum number of boosting trees.
#' @param K_CV Number of group-stratified cross-validation folds, or zero to
#'   fit `num_trees_max` trees without cross-validation. A positive value must
#'   be an integer between two and the smaller group size. CV selects the
#'   minimum mean held-out balancing loss.
#' @details Boosting uses uniformly spaced candidate cuts; Bayesian cut proposals
#'   are uniform. Candidate cuts are relative to the current node. Splits requiring
#'   children with too few observations from either group are excluded. Under
#'   complete separation this restriction can prevent informative splits.
#'   Forward-stagewise fits include a multiplicative normalization constant.
#' @return A list containing `balance_weight_boosting_data` (training weights),
#'   `tree_list` (fitted trees), `c` (normalization), `data_info` (training
#'   domain and group sizes), and `Omega` (domain bounds). With CV, also
#'   returns `loss_CV_store` (folds by candidate tree count).
#' @seealso [batts()], [eval_balance_weight()]
#' @examples
#' set.seed(1)
#' x <- matrix(c(rnorm(30), rnorm(30, 0.5)), ncol = 1)
#' g <- rep(0:1, each = 30)
#' fit <- boots(x, g, num_trees_max = 5, quiet = TRUE)
#' head(2 * log(fit$balance_weight_boosting_data))
#' @md
#' @export
boots = function(data,
                     group_labels,
                     num_trees_max = 100,
                     K_CV = 0 ,
                     max_resol = 4,
                     learn_rate = 0.01,
                     n_bins = 32,
                     n_min_obs_per_node = 1,
                     n_ratio_per_node = 1e-100,
                     margin_scale = 0.1,
                     use_gradient = FALSE,
                     quiet = FALSE
                     ){

  if(!is.numeric(n_ratio_per_node) || length(n_ratio_per_node) != 1L ||
     !is.finite(n_ratio_per_node) || n_ratio_per_node != 1e-100){
    stop("n_ratio_per_node is not implemented; use its default value")
  }
  .validate_group_labels(data, group_labels)

  #Re-scale the data
  d = ncol(data)
  min_max_values = matrix(NA, nrow = d, ncol = 2)

  if(margin_scale >= 0){
    for(j in 1:d){
      min_j = min(data[,j])
      max_j = max(data[,j])
      if(max_j == min_j){
        stop(sprintf("Column %d is constant; remove constant columns before fitting", j))
      }

      margin_size = (max_j - min_j) * margin_scale
      min_j_new = min_j - margin_size
      max_j_new = max_j + margin_size

      data[,j] = (data[,j] - min_j_new) / (max_j_new - min_j_new)
      min_max_values[j,1] = min_j_new
      min_max_values[j,2] = max_j_new
    }
  }else{
    min_max_values[,1] = 0
    min_max_values[,2] = 1
  }


  #obtain the information of the data
  n0 = sum(group_labels == 0)
  n1 = sum(group_labels == 1)
  if(!is.numeric(K_CV) || length(K_CV) != 1L || !is.finite(K_CV) ||
     K_CV != floor(K_CV) ||
     (K_CV != 0 && (K_CV < 2 || K_CV > min(n0, n1)))){
    stop("K_CV must be 0 or an integer between 2 and min(n0, n1)")
  }

  data_info = list("n0" = n0,
                   "n1" = n1,
                   "d" = d,
                   "min_max_values" = min_max_values,
                   "training_data" = data)

  num_trees_opt = num_trees_max

  # make candidates for the cut points
  L_candidates_unif = seq(1/n_bins, 1 - 1/n_bins, by = 1/n_bins)
  y = L_candidates_unif / (1-L_candidates_unif)
  L_candidates = y^(1/1) / (1 + y^(1/1))

  # inputs for the Bayes method
  # not used, only for avoiding errors
  size_burnin = 0
  size_backfitting = 0
  thin = 1
  prob_moves = c(1/4, 1/4, 1/2)
  lambda_0 = 10
  omega_prior_parameters = c(1,1)
  tree_priors = c(0.95,2.0)
  output_BART_ensembles = FALSE

  # optimize the number of trees if the option is yes
  if(K_CV > 0){

    labels_CV = integer(length(group_labels))
    labels_CV[which(group_labels == 0)] = sample(rep(1:K_CV, length.out = sum(group_labels == 0)))
    labels_CV[which(group_labels == 1)] = sample(rep(1:K_CV, length.out = sum(group_labels == 1)))
    loss_CV_store = matrix(NA, nrow = K_CV, ncol = num_trees_max)

    for(k in 1:K_CV){

      if(!quiet){
        print(paste("CV :", k, "/", K_CV), sep = "")
      }

      labels_train = numeric(length(group_labels))
      labels_train[which(labels_CV != k)] = 1

      # boosting
      out_CV = run_adaboost(data,
                         group_labels,
                         num_trees_max,
                         max_resol,
                         learn_rate,
                         L_candidates,
                         labels_train,
                         n_min_obs_per_node,
                         n_ratio_per_node,
                         use_gradient,
                         0,
                         0,
                         1,
                         prob_moves,
                         lambda_0,
                         omega_prior_parameters[1],
                         omega_prior_parameters[2],
                         tree_priors[1],
                         tree_priors[2],
                         FALSE,
                         TRUE
      )

      loss_CV_store[k,] = out_CV$loss_curve
    }

    num_trees_opt = which.min(colMeans(loss_CV_store))

  }

  #Run boosting
  out = run_adaboost(data,
                 group_labels,
                 num_trees_opt,
                 max_resol,
                 learn_rate,
                 L_candidates,
                 rep(1, length(group_labels)),
                 n_min_obs_per_node,
                 n_ratio_per_node,
                 use_gradient,
                 size_burnin,
                 size_backfitting,
                 thin,
                 prob_moves,
                 lambda_0,
                 omega_prior_parameters[1],
                 omega_prior_parameters[2],
                 tree_priors[1],
                 tree_priors[2],
                 output_BART_ensembles,
                 quiet
  )

  out$data_info = data_info

  if(K_CV > 0){
    out$loss_CV_store = loss_CV_store
  }

  out$Omega = min_max_values

  return(out)
}

#' Bayesian additive trees for a two-sample density ratio
#'
#' Draw generalized Bayesian balancing weights w = sqrt(p/q) by backfitting.
#'
#' @inheritParams boots
#' @param num_trees Number of trees in the Bayesian ensemble.
#' @param size_burnin Number of discarded backfitting iterations. Defaults to
#'   `floor(size_backfitting / 2)`; supply a nonnegative integer.
#' @param size_backfitting Required positive integer giving the number of saved
#'   draws. This must be specified explicitly.
#' @param thin Positive integer giving the number of backfitting sweeps per
#'   iteration, including burn-in iterations.
#' @param prob_moves Probabilities of GROW, PRUNE, and CHANGE, in that order;
#'   supply three finite positive values summing to one (tolerance 1e-12).
#' @param lambda_0 Positive leaf-prior scale per tree; ensemble scale is
#'   `num_trees * lambda_0` and remains fixed during sampling.
#' @param omega_prior_parameters Shape and rate of the Gamma prior on the
#'   loss temperature, as a positive length-two vector.
#' @param tree_priors Two tree-depth prior parameters: split probability at
#'   depth d is `tree_priors[1] / (1 + d)^tree_priors[2]`.
#' @param output_BART_ensembles Logical; save posterior forests for subsequent
#'   evaluation on new points. Training-point draws are returned in either case.
#' @details Each fit starts with unsplit trees (`max_resol = 0`). Use multiple
#'   chains and convergence diagnostics for substantive applications. A short
#'   successful run does not establish convergence or frequentist coverage.
#'   Sparse overlap and complete separation can produce prior-sensitive
#'   estimates and slow mixing. The returned intervals are pointwise;
#'   zero inclusion does not establish equality of distributions.
#' @return A list containing `balance_weight_BART_data`, an observations-by-draws
#'   matrix; `forest_list`, saved forests when requested; `omega_store`,
#'   temperature draws; and `lambda_store`, fixed leaf-scale values. Also includes
#'   `tree_list`, `c`, `data_info`, and `Omega` for initialization and evaluation.
#' @seealso [boots()], [eval_balance_weight()]
#' @examples
#' set.seed(2)
#' x <- matrix(c(rnorm(20), rnorm(20, 0.5)), ncol = 1)
#' g <- rep(0:1, each = 20)
#' # Small API demonstration; these settings are not a convergence recommendation.
#' fit <- batts(x, g, num_trees = 4, size_burnin = 5,
#'              size_backfitting = 10, output_BART_ensembles = TRUE, quiet = TRUE)
#' dim(fit$balance_weight_BART_data)
#' @md
#' @export
batts = function(data,
                                     group_labels,
                                     num_trees = 100,
                                     max_resol = 0,
                                     learn_rate = 0.01,
                                     n_bins = 100, # this parameter is to be removed
                                     n_min_obs_per_node = 1,
                                     n_ratio_per_node = 1e-100,
                                     margin_scale = 0.1,
                                     use_gradient = FALSE,
                                     size_burnin = NULL,
                                     size_backfitting = NULL,
                                     thin = 1,
                                     prob_moves = c(1/3,1/3,1/3),
                                     lambda_0 = 5,
                                     omega_prior_parameters = c(1,1),
                                     tree_priors = c(0.95,2.0),
                                     output_BART_ensembles = FALSE,
                                     quiet = FALSE
){

  if(!is.numeric(thin) || length(thin) != 1L || !is.null(dim(thin)) ||
     !is.finite(thin) || thin < 1 || thin != floor(thin) ||
     thin > .Machine$integer.max){
    stop("thin must be a positive integer within the supported integer range")
  }
  if(!is.numeric(size_backfitting) || length(size_backfitting) != 1L ||
     !is.null(dim(size_backfitting)) ||
     !is.finite(size_backfitting) || size_backfitting <= 0 ||
     size_backfitting > .Machine$integer.max ||
     size_backfitting != floor(size_backfitting)){
    stop("size_backfitting must be supplied as a positive integer for batts()")
  }
  if(!is.numeric(n_ratio_per_node) || length(n_ratio_per_node) != 1L ||
     !is.finite(n_ratio_per_node) || n_ratio_per_node != 1e-100){
    stop("n_ratio_per_node is not implemented; use its default value")
  }
  if(!is.numeric(max_resol) || length(max_resol) != 1L ||
     is.na(max_resol) || max_resol != 0){
    stop("batts() requires max_resol = 0 so every initial tree is unsplit")
  }
  .validate_group_labels(data, group_labels)

  if(is.null(size_burnin)){
    size_burnin = floor(size_backfitting / 2)
  }
  if(!is.numeric(size_burnin) || length(size_burnin) != 1L ||
     !is.null(dim(size_burnin)) || !is.finite(size_burnin) ||
     size_burnin < 0 || size_burnin != floor(size_burnin) ||
     size_burnin > .Machine$integer.max){
    stop("size_burnin must be a nonnegative integer within the supported integer range")
  }
  if(!is.numeric(prob_moves) || length(prob_moves) != 3L ||
     !is.null(dim(prob_moves)) || any(!is.finite(prob_moves)) ||
     any(prob_moves <= 0) || abs(sum(prob_moves) - 1) > 1e-12){
    stop("prob_moves must contain three finite positive probabilities summing to one")
  }

  #Re-scale the data
  d = ncol(data)
  min_max_values = matrix(NA, nrow = d, ncol = 2)

  if(margin_scale >= 0){
    for(j in 1:d){
      min_j = min(data[,j])
      max_j = max(data[,j])
      if(max_j == min_j){
        stop(sprintf("Column %d is constant; remove constant columns before fitting", j))
      }

      margin_size = (max_j - min_j) * margin_scale
      min_j_new = min_j - margin_size
      max_j_new = max_j + margin_size

      data[,j] = (data[,j] - min_j_new) / (max_j_new - min_j_new)
      min_max_values[j,1] = min_j_new
      min_max_values[j,2] = max_j_new
    }
  }else{
    min_max_values[,1] = 0
    min_max_values[,2] = 1
  }


  #obtain the information of the data
  n0 = sum(group_labels == 0)
  n1 = sum(group_labels == 1)

  data_info = list("n0" = n0,
                   "n1" = n1,
                   "d" = d,
                   "min_max_values" = min_max_values,
                   "training_data" = data)

  L_candidates_unif = seq(1/n_bins, 1 - 1/n_bins, by = 1/n_bins)
  y = L_candidates_unif / (1-L_candidates_unif)
  L_candidates = y^(1/1) / (1 + y^(1/1))

  #Run the preliminary boosting and the back-fitting method
  out = run_adaboost(data,
                     group_labels,
                     num_trees,
                     max_resol,
                     learn_rate,
                     L_candidates,
                     rep(1, length(group_labels)),
                     n_min_obs_per_node,
                     n_ratio_per_node,
                     use_gradient,
                     size_burnin,
                     size_backfitting,
                     thin,
                     prob_moves,
                     lambda_0,
                     omega_prior_parameters[1],
                     omega_prior_parameters[2],
                     tree_priors[1],
                     tree_priors[2],
                     output_BART_ensembles,
                     quiet
  )

  out$data_info = data_info
  out$Omega = min_max_values

  return(out)
}

#' Evaluate fitted balancing weights
#'
#' Evaluate stored trees at points inside the fitted domain. Weights have scale
#' w = sqrt(p/q), with group 0 in the numerator; use `2 * log(w)` for log ratios.
#'
#' @param list_result Result of [boots()] or [batts()].
#' @param eval_points Finite numeric matrix with the same columns, in the same
#'   order, as the training matrix. All points must lie inside `Omega`.
#' @param is_Bayes Logical; additionally evaluate posterior forests. Requires
#'   fitting with `output_BART_ensembles = TRUE`.
#' @return A list with `balancing_weight_boosting`, a one-column numeric matrix,
#'   and, when
#'   `is_Bayes = TRUE`, `balancing_weight_BART`, an evaluation-points-by-draws
#'   matrix. Both include the fitted normalization constant.
#' @details Prediction does not draw random numbers. With `is_Bayes = TRUE`,
#'   the boosting component describes the initialization; the posterior draws
#'   are in `balancing_weight_BART`. Missing posterior forests cause an error.
#' @seealso [boots()], [batts()]
#' @examples
#' set.seed(3)
#' x <- matrix(c(rnorm(20), rnorm(20, 0.5)), ncol = 1)
#' fit <- boots(x, rep(0:1, each = 20), num_trees_max = 5, quiet = TRUE)
#' pred <- eval_balance_weight(fit, x[1:3, , drop = FALSE])
#' pred$balancing_weight_boosting
#' @md
#' @export
eval_balance_weight = function(list_result, eval_points, is_Bayes = FALSE){

  data_info = list_result$data_info
  if(!is.matrix(eval_points) || !is.numeric(eval_points) ||
     ncol(eval_points) != data_info$d || nrow(eval_points) < 1L ||
     any(!is.finite(eval_points))){
    stop("eval_points must be a nonempty finite numeric matrix with the training column count")
  }

  if(length(list_result$tree_list) == 0){

    stop("The tree list is empty")

  }else{
    if(is_Bayes && (is.null(list_result$forest_list) || length(list_result$forest_list) == 0)){
      stop("Posterior forests were not saved; fit with output_BART_ensembles = TRUE")
    }

    #re-scale the input data
    min_max_values = data_info$min_max_values
    d = data_info$d

    for(j in 1:d){
      min_j_new = min_max_values[j,1]
      max_j_new = min_max_values[j,2]

      if(sum(eval_points[,j] < min_j_new) > 0 || sum(eval_points[,j] > max_j_new) > 0){
        stop("Some points are outside of the sample space")
      }

      eval_points[,j] = (eval_points[,j] - min_j_new) / (max_j_new - min_j_new)
    }

    out = list()

    out_temp = evaluate_balance_weight_boosting(list_result$tree_list, eval_points)
    out$balancing_weight_boosting = list_result$c * out_temp$balance_current

    if(is_Bayes){
      out_temp = evaluate_balance_weight_BART(list_result$forest_list, eval_points)
      out$balancing_weight_BART = list_result$c * out_temp$balance_store
    }
  }

  return(out)

}
