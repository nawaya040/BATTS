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
  cat("  estimand:     balancing weight w = sqrt(p/q), group 0 in the numerator\n")
  cat("                2 * log(w) is the log-density ratio\n")
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
  cat("  estimand:     balancing weight w = sqrt(p/q), group 0 in the numerator\n")
  cat("                2 * log(w) is the log-density ratio\n")
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
  cat("  fitted weight w = sqrt(p/q) range: [",
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

#' Predict balancing weights and density ratios
#'
#' @param object A `batts_boost` or `batts_bart` fit.
#' @param newdata Finite numeric matrix with the same columns, in the same
#'   order, as the training matrix. All points must lie inside the fitted
#'   domain. If omitted, the training observations are used.
#' @param type `"log_ratio"`, the default, returns the log-density ratio
#'   `2 * log(w) = log(p/q)`. `"ratio"` returns `w^2 = p/q`. `"weight"` returns
#'   the balancing weight `w = sqrt(p/q)` itself, exactly as
#'   [eval_balance_weight()] does. Group 0 is in the numerator throughout.
#' @param summary For a `batts_bart` fit: `"mean"` or `"median"` collapses the
#'   posterior draws and `"none"` returns the points-by-draws matrix. The
#'   transformation implied by `type` is applied to each draw before
#'   summarizing, so the result summarizes the posterior of the transformed
#'   quantity.
#' @param level Credible level used when `interval = TRUE`.
#' @param interval For a `batts_bart` fit, return a matrix with columns `fit`,
#'   `lower` and `upper` instead of a vector.
#' @param ... Not used. A `batts_boost` fit has no posterior draws, so passing
#'   `summary`, `level` or `interval` to it is an error rather than silently
#'   ignored.
#' @return A numeric vector, or a matrix when `summary = "none"` or
#'   `interval = TRUE`.
#' @details Prediction draws no random numbers.
#' @name batts-predict
#' @md
NULL

.batts_transform = function(w, type){
  switch(type,
         log_ratio = 2 * log(w),
         ratio     = w^2,
         weight    = w)
}

.batts_no_extra_args = function(n, dots){
  if(n > 0L){
    nm = names(dots)
    nm = if(is.null(nm)) character(0) else nm[nzchar(nm)]
    stop("a boosting fit has no posterior draws; unused argument",
         if(n > 1L) "s" else "",
         if(length(nm) > 0L) paste0(": ", paste(nm, collapse = ", ")) else "")
  }
  invisible(NULL)
}

#' @rdname batts-predict
#' @importFrom stats predict
#' @export
predict.batts_boost = function(object, newdata,
                               type = c("log_ratio", "ratio", "weight"), ...){
  type = match.arg(type)
  .batts_no_extra_args(...length(), list(...))
  w = if(missing(newdata)){
    as.numeric(object$balance_weight_boosting_data)
  }else{
    as.numeric(eval_balance_weight(object, newdata)$balancing_weight_boosting)
  }
  .batts_transform(w, type)
}

#' @rdname batts-predict
#' @importFrom stats predict
#' @export
predict.batts_bart = function(object, newdata,
                              type = c("log_ratio", "ratio", "weight"),
                              summary = c("mean", "median", "none"),
                              level = 0.95, interval = FALSE, ...){
  type = match.arg(type); summary = match.arg(summary)
  if(!is.numeric(level) || length(level) != 1L || !is.finite(level) ||
     level <= 0 || level >= 1){
    stop("level must be a single number strictly between zero and one")
  }
  if(!is.logical(interval) || length(interval) != 1L || is.na(interval)){
    stop("interval must be TRUE or FALSE")
  }
  if(!missing(newdata) && length(object$forest_list) == 0){
    stop("Posterior forests were not saved, so the fit cannot be evaluated at new points.\n",
         "Refit with bart_control(keep_draws = TRUE) ",
         "(or output_BART_ensembles = TRUE in batts()).\n",
         "fitted() and predict() without newdata still work.", call. = FALSE)
  }
  w = if(missing(newdata)){
    object$balance_weight_BART_data
  }else{
    eval_balance_weight(object, newdata, is_Bayes = TRUE)$balancing_weight_BART
  }
  draws = .batts_transform(w, type)
  if(interval){
    a = (1 - level) / 2
    return(cbind(
      fit   = if(summary == "median") apply(draws, 1, stats::median)
              else rowMeans(draws),
      lower = apply(draws, 1, stats::quantile, probs = a, names = FALSE),
      upper = apply(draws, 1, stats::quantile, probs = 1 - a, names = FALSE)))
  }
  switch(summary,
         none   = draws,
         median = apply(draws, 1, stats::median),
         mean   = rowMeans(draws))
}

#' @rdname batts-predict
#' @importFrom stats fitted
#' @export
fitted.batts_boost = function(object, type = c("log_ratio", "ratio", "weight"), ...){
  type = match.arg(type)
  .batts_no_extra_args(...length(), list(...))
  .batts_transform(as.numeric(object$balance_weight_boosting_data), type)
}

#' @rdname batts-predict
#' @importFrom stats fitted
#' @export
fitted.batts_bart = function(object, type = c("log_ratio", "ratio", "weight"),
                             summary = c("mean", "median", "none"), ...){
  type = match.arg(type); summary = match.arg(summary)
  draws = .batts_transform(object$balance_weight_BART_data, type)
  switch(summary,
         none   = draws,
         median = apply(draws, 1, stats::median),
         mean   = rowMeans(draws))
}

#' Density ratio of a fitted BATTS model
#'
#' The density ratio `p/q` with group 0 in the numerator, that is `w^2` for the
#' fitted balancing weight `w`. Equivalent to
#' `predict(object, newdata, type = "ratio")`.
#'
#' @param object A `batts_boost` or `batts_bart` fit.
#' @param newdata Evaluation points; see [batts-predict]. If omitted, the
#'   training observations are used.
#' @param ... Passed to the `predict` method.
#' @return A numeric vector, or a matrix for a `batts_bart` fit with
#'   `summary = "none"` or `interval = TRUE`.
#' @seealso [batts-predict]
#' @md
#' @export
density_ratio = function(object, newdata, ...) UseMethod("density_ratio")

#' @rdname density_ratio
#' @export
density_ratio.batts_fit = function(object, newdata, ...){
  if(missing(newdata)) predict(object, type = "ratio", ...)
  else predict(object, newdata, type = "ratio", ...)
}
