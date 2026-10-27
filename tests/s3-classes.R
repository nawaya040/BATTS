library(BATTS)
# S3 classes and display methods (BATTS 0.3.0.9000).
set.seed(21)
x <- matrix(c(rnorm(30), rnorm(30, 0.5)), ncol = 1); g <- rep(0:1, each = 30)

set.seed(3); fb <- boots(x, g, num_trees_max = 6, K_CV = 3, quiet = TRUE)
set.seed(4); fs <- batts(x, g, num_trees = 4, size_burnin = 5, size_backfitting = 8,
                         output_BART_ensembles = TRUE, quiet = TRUE)

# Class structure: BART does not inherit the boosting class.
stopifnot(inherits(fb, "batts_boost"), inherits(fb, "batts_fit"),
          inherits(fs, "batts_bart"),  inherits(fs, "batts_fit"),
          !inherits(fs, "batts_boost"),
          !inherits(fb, "batts_bart"))

# Nothing but the class attribute changed.
stopifnot(identical(class(unclass(fb)), "list"),
          identical(class(unclass(fs)), "list"),
          identical(names(unclass(fb)),
                    c("balance_weight_boosting_data", "n_nodes", "loss_curve",
                      "tree_list", "c", "data_info", "loss_CV_store", "Omega")))

# Display methods run and return their argument invisibly.
out <- capture.output({
  stopifnot(identical(withVisible(print(fb))$visible, FALSE),
            identical(withVisible(print(fs))$visible, FALSE))
})
stopifnot(any(grepl("BATTS boosting fit", out, fixed = TRUE)),
          any(grepl("BATTS Bayesian additive trees fit", out, fixed = TRUE)))

sb <- summary(fb); ss <- summary(fs)
stopifnot(inherits(sb, "summary.batts_fit"), inherits(ss, "summary.batts_fit"),
          identical(sb$type, "boost"), identical(ss$type, "bart"),
          sb$n0 == 30, sb$n1 == 30, sb$d == 1,
          ss$draws == 8, isTRUE(ss$forests_saved),
          sb$num_trees == length(fb$tree_list),
          length(sb$weight_range) == 2, all(is.finite(sb$weight_range)),
          length(ss$weight_range) == 2, all(is.finite(ss$weight_range)))

out2 <- capture.output({
  stopifnot(identical(withVisible(print(sb))$visible, FALSE),
            identical(withVisible(print(ss))$visible, FALSE))
})
stopifnot(any(grepl("fitted density ratio range", out2, fixed = TRUE)))

stopifnot(nobs(fb) == 60, nobs(fs) == 60, identical(nobs(fb), nobs(fs)))

# The methods consume no random numbers.
rng <- .Random.seed
invisible(capture.output({print(fb); print(fs); print(summary(fb)); print(summary(fs))}))
invisible(nobs(fb))
stopifnot(identical(rng, .Random.seed))

cat('S3 class checks passed.\n')
