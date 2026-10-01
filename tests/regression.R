library(BATTS)
assert_close <- function(a, b, tolerance = 1e-10) {
  if (!isTRUE(all.equal(a, b, tolerance = tolerance, check.attributes = FALSE)))
    stop("Numerical invariant failed")
}
expect_error <- function(expr, pattern) {
  message <- tryCatch({ force(expr); NA_character_ }, error = conditionMessage)
  if (is.na(message) || !grepl(pattern, message, fixed = TRUE))
    stop("Expected error containing: ", pattern)
}
set.seed(1201)
x <- cbind(c(rnorm(32, -0.3), rnorm(24, 0.4)), rnorm(56))
g <- c(rep(0, 32), rep(1, 24))
for (gradient in c(FALSE, TRUE)) {
  fit_one <- function() boots(x, g, num_trees_max = 12, K_CV = 4,
                             max_resol = 2, n_bins = 8,
                             use_gradient = gradient, quiet = TRUE)
  set.seed(701)
  fit <- fit_one()
  fit_rng <- .Random.seed
  pred <- eval_balance_weight(fit, x)$balancing_weight_boosting
  stopifnot(identical(fit_rng, .Random.seed), all(is.finite(pred)), all(pred > 0))
  assert_close(pred, fit$balance_weight_boosting_data)
  set.seed(701)
  again <- fit_one()
  stopifnot(identical(fit, again), identical(fit_rng, .Random.seed))
  stopifnot(ncol(fit$loss_CV_store) == 12,
            all(is.finite(fit$loss_CV_store)))
}
# Permuted labels and small minority groups must remain valid under CV.
ix <- as.vector(rbind(1:24, 33:56))
ix <- c(ix, 25:32)
set.seed(702)
u <- boots(x[ix, ], g[ix], num_trees_max = 5, K_CV = 4,
           n_bins = 8, max_resol = 2, use_gradient = TRUE, quiet = TRUE)
stopifnot(all(is.finite(u$loss_CV_store)))
# Check the prediction boundary independently with a hand-specified tree.
manual <- list(tree_list = list(list(d = c(0L, -1L, -1L),
                                    l = c(0.5, 0, 0), beta = c(1, 4, 9))),
               c = 1.7, data_info = list(d = 1L, min_max_values = matrix(c(0, 1), 1)))
set.seed(703)
rng <- .Random.seed
edge <- eval_balance_weight(manual, matrix(c(0, 0.5 - 1e-8, 0.5, 1), ncol = 1))
assert_close(as.numeric(edge$balancing_weight_boosting), 1.7 * c(2, 2, 3, 3))
stopifnot(identical(rng, .Random.seed))
expect_error(eval_balance_weight(manual, matrix(1.01, 1)), "outside")
# Exercise actual fitted forests with repeated values at candidate cuts.
tied_x <- matrix(rep(seq(0, 1, by = 0.125), each = 4), ncol = 1)
tied_g <- rep(c(0, 0, 1, 1), 9)
for (gradient in c(FALSE, TRUE)) {
  tfit <- boots(tied_x, tied_g, num_trees_max = 4, margin_scale = -1,
                n_bins = 8, use_gradient = gradient, quiet = TRUE)
  assert_close(eval_balance_weight(tfit, tied_x)$balancing_weight_boosting,
               tfit$balance_weight_boosting_data)
}
fit_bayes <- function(save = TRUE) batts(x, g, num_trees = 8,
  size_burnin = 15, size_backfitting = 30, output_BART_ensembles = save,
  quiet = TRUE)
set.seed(704)
b <- fit_bayes()
brng <- .Random.seed
stopifnot(identical(dim(b$balance_weight_BART_data), c(56L, 30L)),
          length(b$forest_list) == 30L,
          all(is.finite(b$balance_weight_BART_data)),
          all(b$balance_weight_BART_data > 0))
for (i in seq_len(15)) {
  bp <- eval_balance_weight(b, x, is_Bayes = TRUE)$balancing_weight_BART
  assert_close(bp, b$balance_weight_BART_data)
}
stopifnot(identical(brng, .Random.seed))
set.seed(704)
b2 <- fit_bayes()
stopifnot(identical(b, b2), identical(brng, .Random.seed))
set.seed(704)
b_no <- fit_bayes(FALSE)
assert_close(b_no$balance_weight_BART_data, b$balance_weight_BART_data, 0)
stopifnot(identical(brng, .Random.seed))
expect_error(eval_balance_weight(b_no, x, is_Bayes = TRUE), "Posterior forests were not saved")
expect_error(batts(x, g), "size_backfitting must be supplied")
expect_error(batts(x, g, size_backfitting = 2, max_resol = 1), "max_resol = 0")
expect_error(boots(x, g + 1), "group_labels")
expect_error(boots(x, rep(0, nrow(x))), "group_labels")
expect_error(boots(x, g[-1]), "group_labels")
expect_error(boots(cbind(x, 1), g), "constant")
expect_error(batts(cbind(x, 1), g, size_backfitting = 2), "constant")
expect_error(boots(x, g, K_CV = 1), "K_CV")
expect_error(boots(x, g, n_ratio_per_node = 0.1), "not implemented")
cat("BATTS numerical regression tests passed.\n")
