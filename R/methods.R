#' Methods for BATTS fits
#'
#' Display and summary methods for the objects returned by [boots()] and
#' [batts()].
#'
#' @param x,object A `batts_boost` or `batts_bart` fit, or a
#'   `summary.batts_fit` object.
#' @param ... Not used.
#' @return `print` returns its argument invisibly. `summary` returns an object
#'   of class `summary.batts_fit`. `nobs` returns the number of training
#'   observations.
#' @name batts-methods
#' @md
NULL

#' @rdname batts-methods
#' @export
print.batts_boost = function(x, ...){
  info = x$data_info
  cat("<BATTS boosting fit>\n")
  cat("  observations: ", info$n0 + info$n1,
      " (group 0: ", info$n0, ", group 1: ", info$n1, ")\n", sep = "")
  cat("  variables:    ", info$d, "\n", sep = "")
  cat("  trees:        ", length(x$tree_list), "\n", sep = "")
  if(is.null(x$loss_CV_store)){
    cat("  CV:           not used\n")
  }else{
    cat("  CV:           ", nrow(x$loss_CV_store), " folds, ",
        ncol(x$loss_CV_store), " candidate tree counts\n", sep = "")
  }
  cat("  estimand:     density ratio of group 1 over group 0\n")
  invisible(x)
}

#' @rdname batts-methods
#' @export
print.batts_bart = function(x, ...){
  info = x$data_info
  cat("<BATTS Bayesian additive trees fit>\n")
  cat("  observations: ", info$n0 + info$n1,
      " (group 0: ", info$n0, ", group 1: ", info$n1, ")\n", sep = "")
  cat("  variables:    ", info$d, "\n", sep = "")
  cat("  saved draws:  ", ncol(x$balance_weight_BART_data), "\n", sep = "")
  cat("  forests kept: ", if(length(x$forest_list) > 0) "yes" else "no",
      "\n", sep = "")
  cat("  boosting init trees: ", length(x$tree_list), "\n", sep = "")
  cat("  estimand:     density ratio of group 1 over group 0\n")
  invisible(x)
}

#' @rdname batts-methods
#' @export
summary.batts_fit = function(object, ...){
  info = object$data_info
  out = list(n0 = info$n0, n1 = info$n1, d = info$d,
             domain = info$min_max_values,
             num_trees = length(object$tree_list))
  if(inherits(object, "batts_bart")){
    out$type = "bart"
    out$draws = ncol(object$balance_weight_BART_data)
    out$forests_saved = length(object$forest_list) > 0
    weights = rowMeans(object$balance_weight_BART_data)
  }else{
    out$type = "boost"
    out$cv = object$loss_CV_store
    weights = as.numeric(object$balance_weight_boosting_data)
  }
  out$weight_range = range(weights)
  class(out) = "summary.batts_fit"
  out
}

#' @rdname batts-methods
#' @export
print.summary.batts_fit = function(x, ...){
  cat(if(x$type == "bart") "<BATTS Bayesian additive trees fit>\n"
      else "<BATTS boosting fit>\n")
  cat("  observations: ", x$n0 + x$n1,
      " (group 0: ", x$n0, ", group 1: ", x$n1, ")\n", sep = "")
  cat("  variables:    ", x$d, "\n", sep = "")
  cat("  trees:        ", x$num_trees, "\n", sep = "")
  if(x$type == "bart"){
    cat("  saved draws:  ", x$draws, "\n", sep = "")
    cat("  forests kept: ", if(x$forests_saved) "yes" else "no", "\n", sep = "")
  }else if(!is.null(x$cv)){
    cat("  CV:           ", nrow(x$cv), " folds, ", ncol(x$cv),
        " candidate tree counts\n", sep = "")
  }
  cat("  training domain (min, max) per variable:\n")
  print(x$domain)
  cat("  fitted density ratio range: [",
      format(x$weight_range[1], digits = 6), ", ",
      format(x$weight_range[2], digits = 6), "]\n", sep = "")
  invisible(x)
}

#' @rdname batts-methods
#' @importFrom stats nobs
#' @export
nobs.batts_fit = function(object, ...){
  object$data_info$n0 + object$data_info$n1
}
