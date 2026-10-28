library(BATTS)
# boots(), batts() and eval_balance_weight() are deprecated aliases: they warn
# once per session and return exactly what the internal functions return.
set.seed(61)
x <- matrix(c(rnorm(30), rnorm(30, 0.5)), ncol = 1); g <- rep(0:1, each = 30)
nd <- x[1:3, , drop = FALSE]
warnings_of <- function(expr){
  w <- list()
  value <- withCallingHandlers(expr, warning = function(c){
    w[[length(w) + 1L]] <<- c; invokeRestart("muffleWarning")})
  list(value = value, warnings = w)
}
core <- function(f) unclass(f)[setdiff(names(f), c("call", "control", "group_levels",
                                                    "variable_names"))]

# 1. The wrappers keep the signatures and defaults of the internal functions.
stopifnot(identical(formals(boots), formals(BATTS:::.fit_boost)),
          identical(formals(batts), formals(BATTS:::.fit_bart)),
          identical(formals(eval_balance_weight), formals(BATTS:::.eval_weights)))

# 2. The new API never warns.
w_new <- warnings_of({
  set.seed(3); fb <- batts_boost(x, g, control = boost_control(num_trees_max = 6, K_CV = 3),
                                 quiet = TRUE); rb <- .Random.seed
  set.seed(4); fs <- batts_bart(x, g, control = bart_control(num_trees = 4, burnin = 5,
                                draws = 8, keep_draws = TRUE), quiet = TRUE); rs <- .Random.seed
  list(predict(fb, nd), predict(fs, nd), fitted(fb), density_ratio(fs, nd))
})
stopifnot(length(w_new$warnings) == 0L)

# 3. Each deprecated function warns once, with class deprecatedWarning, and
#    returns the same object and random number state as the new API.
w1 <- warnings_of({
  set.seed(3); ob <- boots(x, g, num_trees_max = 6, K_CV = 3, quiet = TRUE); rb2 <- .Random.seed
  set.seed(4); os <- batts(x, g, num_trees = 4, size_burnin = 5, size_backfitting = 8,
                           output_BART_ensembles = TRUE, quiet = TRUE); rs2 <- .Random.seed
  eb <- eval_balance_weight(fb, nd)
  es <- eval_balance_weight(fs, nd, is_Bayes = TRUE)
})
msgs <- vapply(w1$warnings, conditionMessage, character(1))
stopifnot(length(w1$warnings) == 3L,
          all(vapply(w1$warnings, inherits, logical(1), "deprecatedWarning")),
          grepl("boots() is deprecated; use batts_boost()", msgs[1], fixed = TRUE),
          grepl("batts() is deprecated; use batts_bart()", msgs[2], fixed = TRUE),
          grepl("eval_balance_weight() is deprecated; use predict()", msgs[3], fixed = TRUE),
          identical(unclass(ob), core(fb)), identical(rb2, rb),
          identical(unclass(os), core(fs)), identical(rs2, rs),
          identical(as.numeric(eb$balancing_weight_boosting), predict(fb, nd, type = "weight")),
          identical(es$balancing_weight_BART, predict(fs, nd, type = "weight", summary = "none")))

# 4. Later calls in the same session do not warn again.
w2 <- warnings_of({
  boots(x, g, num_trees_max = 2, K_CV = 0, quiet = TRUE)
  batts(x, g, num_trees = 2, size_burnin = 1, size_backfitting = 2, quiet = TRUE)
  eval_balance_weight(fb, nd)
})
stopifnot(length(w2$warnings) == 0L)

cat("Deprecation checks passed.\n")
