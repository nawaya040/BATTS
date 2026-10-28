library(BATTS)
# plot() methods draw without error, return the fit invisibly and use no
# random numbers.
pdf(NULL)
set.seed(71)
x1 <- rnorm(40); y1 <- rnorm(40, 0.5)
x2 <- matrix(rnorm(160), ncol = 2)
bad <- function(expr) tryCatch({expr; NA_character_}, error = conditionMessage)
fb <- batts_boost(x1, y = y1, control = boost_control(num_trees_max = 10, K_CV = 2),
                  quiet = TRUE)
fb0 <- batts_boost(x1, y = y1, control = boost_control(num_trees_max = 5, K_CV = 0),
                   quiet = TRUE)
fs <- batts_bart(x1, y = y1, control = bart_control(num_trees = 4, burnin = 5, draws = 8),
                 quiet = TRUE)
fb2 <- batts_boost(x2, rep(0:1, each = 40),
                   control = boost_control(num_trees_max = 5, K_CV = 0), quiet = TRUE)

seed <- .Random.seed
v <- list(withVisible(plot(fb)), withVisible(plot(fb, which = "fit")),
          withVisible(plot(fb0)), withVisible(plot(fs)),
          withVisible(plot(fs, which = "fit", level = 0.8)),
          withVisible(plot(fb, main = "custom", xlab = "trees", col = "red")))
stopifnot(identical(.Random.seed, seed),
          !any(vapply(v, `[[`, logical(1), "visible")),
          identical(v[[1]]$value, fb), identical(v[[4]]$value, fs),
          grepl("did not use cross-validation", bad(plot(fb0, which = "cv"))),
          grepl("one variable", bad(plot(fb2, which = "fit"))),
          grepl("level must be", bad(plot(fs, which = "fit", level = 2))),
          grepl("should be one of", bad(plot(fs, which = "cv"))))
dev.off()

cat("plot() checks passed.\n")
