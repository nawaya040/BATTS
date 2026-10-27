library(BATTS)
# predict()/fitted()/density_ratio() agree with eval_balance_weight() exactly.
set.seed(31)
x <- matrix(c(rnorm(40), rnorm(40, 0.6)), ncol = 1); g <- rep(0:1, each = 40)
nd <- x[1:5, , drop = FALSE]

set.seed(7); fb <- boots(x, g, num_trees_max = 8, K_CV = 3, quiet = TRUE)
set.seed(8); fs <- batts(x, g, num_trees = 4, size_burnin = 5, size_backfitting = 13,
                         output_BART_ensembles = TRUE, quiet = TRUE)

# 1. type = "weight" is bit identical to the legacy entry point.
legacy_b <- as.numeric(eval_balance_weight(fb, nd)$balancing_weight_boosting)
stopifnot(identical(predict(fb, nd, type = "weight"), legacy_b))

legacy_s <- eval_balance_weight(fs, nd, is_Bayes = TRUE)$balancing_weight_BART
stopifnot(identical(predict(fs, nd, type = "weight", summary = "none"), legacy_s))

# 2. The default is the log-density ratio, not the weight.
stopifnot(identical(predict(fb, nd), 2 * log(legacy_b)),
          !identical(predict(fb, nd), legacy_b))

# 3. Transformations are exact.
w <- predict(fb, nd, type = "weight")
stopifnot(identical(predict(fb, nd, type = "ratio"), w^2),
          identical(predict(fb, nd, type = "log_ratio"), 2 * log(w)),
          identical(density_ratio(fb, nd), w^2))

# 4. Each draw is transformed before summarizing, not the other way round.
d <- predict(fs, nd, type = "weight", summary = "none")
stopifnot(identical(predict(fs, nd, type = "ratio"), rowMeans(d^2)),
          identical(predict(fs, nd), rowMeans(2 * log(d))),
          !isTRUE(all.equal(rowMeans(d^2), rowMeans(d)^2)))

# 5. The median summary is invariant to the monotone transform only when the
#    number of draws is odd, because median() averages the two middle values.
stopifnot(ncol(d) == 13L, ncol(d) %% 2L == 1L)
stopifnot(isTRUE(all.equal(predict(fs, nd, type = "ratio", summary = "median"),
                           apply(d, 1, median)^2, tolerance = 1e-12)))

# 6. fitted() matches prediction at the training points.
stopifnot(identical(fitted(fb, type = "weight"),
                    as.numeric(fb$balance_weight_boosting_data)),
          isTRUE(all.equal(fitted(fb), predict(fb, x), tolerance = 1e-10)),
          identical(fitted(fs, type = "weight", summary = "none"),
                    fs$balance_weight_BART_data))

# 7. Omitting newdata uses the training observations.
stopifnot(identical(predict(fb), fitted(fb)),
          identical(predict(fs), fitted(fs)),
          identical(density_ratio(fb), fitted(fb, type = "ratio")))

# 8. Intervals.
iv <- predict(fs, nd, interval = TRUE, level = 0.8)
stopifnot(identical(dim(iv), c(5L, 3L)),
          identical(colnames(iv), c("fit", "lower", "upper")),
          all(iv[, "lower"] <= iv[, "fit"]), all(iv[, "fit"] <= iv[, "upper"]),
          all(is.finite(iv)))

# 9. Invalid arguments are rejected.
bad <- function(expr) tryCatch({expr; NA_character_}, error = conditionMessage)
stopifnot(grepl("should be one of", bad(predict(fb, nd, type = "nope"))),
          grepl("level must be", bad(predict(fs, nd, level = 1)), fixed = TRUE),
          grepl("interval must be", bad(predict(fs, nd, interval = NA)), fixed = TRUE),
          grepl("outside of the sample space", bad(predict(fb, x + 1e6)), fixed = TRUE))

# 10. A boosting fit rejects posterior-only arguments instead of ignoring them.
stopifnot(grepl("no posterior draws", bad(predict(fb, nd, summary = "none")), fixed = TRUE),
          grepl("summary", bad(predict(fb, nd, summary = "none")), fixed = TRUE),
          grepl("no posterior draws", bad(fitted(fb, interval = TRUE)), fixed = TRUE))

# 11. Prediction consumes no random numbers.
rng <- .Random.seed
invisible(predict(fb, nd)); invisible(predict(fs, nd))
invisible(density_ratio(fb, nd)); invisible(density_ratio(fs, nd))
invisible(fitted(fb)); invisible(fitted(fs))
invisible(predict(fs, nd, interval = TRUE))
stopifnot(identical(rng, .Random.seed))

cat('predict()/fitted()/density_ratio() checks passed.\n')
