#' Plot a BATTS fit
#'
#' Diagnostic and summary plots drawn with base graphics.
#'
#' For a boosting fit, `which = "cv"` plots the cross-validation loss, averaged
#' over folds, against the number of trees and marks the selected number;
#' `which = "fit"` plots the fitted log-density ratio `log(p/q)` at the
#' training points. For a Bayesian additive trees fit, `which = "trace"` plots
#' the saved draws of the loss temperature, and `which = "fit"` plots the
#' posterior mean of `log(p/q)` at the training points with a pointwise
#' credible band.
#'
#' `which = "fit"` needs a fit with one variable. For more variables, evaluate
#' the fit with [predict()] and plot the result.
#'
#' @param x A `batts_boost` or `batts_bart` fit.
#' @param which The plot to draw. For a boosting fit the default is `"cv"`
#'   when cross-validation was used and `"fit"` otherwise; for a Bayesian
#'   additive trees fit it is `"trace"`.
#' @param level Credible level of the band for `which = "fit"`.
#' @param ... Graphical parameters passed to [plot.default()], such as `main`,
#'   `xlab`, `ylab` or `col`; they replace the defaults.
#' @return `x`, invisibly.
#' @details Plotting draws no random numbers.
#' @seealso [batts_boost()], [batts_bart()], [batts-predict]
#' @examples
#' set.seed(1)
#' fit <- batts_boost(rnorm(100), y = rnorm(100, 0.5),
#'                    control = boost_control(num_trees_max = 100, depth = 2),
#'                    quiet = TRUE)
#' plot(fit)
#' plot(fit, which = "fit")
#' @name batts-plot
#' @md
NULL

# Draw with defaults that the caller's graphical parameters replace.
.plot_with = function(x, y, defaults, dots){
  defaults[names(dots)] = dots
  do.call(graphics::plot, c(list(x, y), defaults))
}

# Training points of a one-variable fit on the original scale, in order.
.training_axis = function(object){
  info = object$data_info
  if(info$d != 1L){
    stop("which = \"fit\" plots fits with one variable; use predict() for fits with more",
         call. = FALSE)
  }
  dom = info$min_max_values
  t = info$training_data[, 1] * (dom[1, 2] - dom[1, 1]) + dom[1, 1]
  list(x = t, order = order(t))
}

.ratio_label = function(object){
  lev = object$group_levels
  if(is.null(lev)) "log(p/q), p = group 0, q = group 1"
  else sprintf("log(p/q), p = \"%s\", q = \"%s\"", lev[1], lev[2])
}

#' @rdname batts-plot
#' @export
plot.batts_boost = function(x, which = c("cv", "fit"), ...){
  which = if(missing(which)) (if(is.null(x$loss_CV_store)) "fit" else "cv")
          else match.arg(which)
  dots = list(...)
  if(which == "cv"){
    if(is.null(x$loss_CV_store)){
      stop("this fit did not use cross-validation (K_CV = 0); use which = \"fit\"",
           call. = FALSE)
    }
    loss = colMeans(x$loss_CV_store)
    selected = length(x$tree_list)
    .plot_with(seq_along(loss), loss,
               list(type = "l", xlab = "number of trees",
                    ylab = "mean held-out loss",
                    main = sprintf("Cross-validation (%d folds)",
                                   nrow(x$loss_CV_store))), dots)
    graphics::abline(v = selected, lty = 2)
    graphics::points(selected, loss[selected], pch = 19)
  }else{
    ax = .training_axis(x)
    o = ax$order
    .plot_with(ax$x[o], fitted(x)[o],
               list(type = "s", xlab = "x", ylab = .ratio_label(x),
                    main = "Boosting fit"), dots)
    graphics::abline(h = 0, lty = 3)
    graphics::rug(ax$x)
  }
  invisible(x)
}

#' @rdname batts-plot
#' @export
plot.batts_bart = function(x, which = c("trace", "fit"), level = 0.95, ...){
  which = match.arg(which)
  dots = list(...)
  if(which == "trace"){
    omega = as.numeric(x$omega_store)
    .plot_with(seq_along(omega), omega,
               list(type = "l", xlab = "saved draw",
                    ylab = "loss temperature", main = "Trace of the temperature"),
               dots)
  }else{
    if(!is.numeric(level) || length(level) != 1L || !is.finite(level) ||
       level <= 0 || level >= 1){
      stop("level must be a single number strictly between zero and one",
           call. = FALSE)
    }
    ax = .training_axis(x)
    o = ax$order
    draws = fitted(x, summary = "none")[o, , drop = FALSE]
    a = (1 - level) / 2
    lower = apply(draws, 1, stats::quantile, probs = a, names = FALSE)
    upper = apply(draws, 1, stats::quantile, probs = 1 - a, names = FALSE)
    .plot_with(range(ax$x), range(lower, upper),
               list(type = "n", xlab = "x", ylab = .ratio_label(x),
                    main = sprintf("Posterior mean and %g%% pointwise band",
                                   100 * level)), dots)
    graphics::polygon(c(ax$x[o], rev(ax$x[o])), c(lower, rev(upper)),
                      col = "grey85", border = NA)
    graphics::lines(ax$x[o], rowMeans(draws))
    graphics::abline(h = 0, lty = 3)
    graphics::rug(ax$x)
  }
  invisible(x)
}
