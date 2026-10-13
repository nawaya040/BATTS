library(BATTS)
# Defaults equal the paper's simulation settings (BATTS 0.3.0).
b <- formals(boots); s <- formals(batts)
stopifnot(identical(b$num_trees_max, 1000), identical(b$K_CV, 5), identical(b$max_resol, 4),
          identical(b$learn_rate, 0.01), identical(b$n_bins, 32), identical(b$margin_scale, 0.1),
          identical(b$n_min_obs_per_node, 1), isTRUE(b$use_gradient),
          identical(b$subsample_fraction, 0.5),
          identical(s$num_trees, 200), identical(s$size_burnin, 2000),
          identical(s$size_backfitting, 1000), identical(s$lambda_0, 5), identical(s$thin, 1),
          identical(eval(s$prob_moves), c(1/3, 1/3, 1/3)), identical(eval(s$omega_prior_parameters), c(1, 1)),
          identical(eval(s$tree_priors), c(0.95, 2)), identical(s$margin_scale, 0.1),
          identical(s$n_bins, 100), identical(s$max_resol, 0), isFALSE(s$use_gradient))
# A default boots() call equals the explicit simulation call with the same seed.
set.seed(5)
x <- cbind(c(rnorm(40), rnorm(40, .6)), rnorm(80)); g <- rep(0:1, each = 40)
set.seed(11); d <- boots(x, g, num_trees_max = 30, quiet = TRUE)
set.seed(11); e <- boots(x, g, num_trees_max = 30, K_CV = 5, max_resol = 4, learn_rate = .01,
                         use_gradient = TRUE, subsample_fraction = .5, quiet = TRUE)
stopifnot(identical(d, e))
cat('Default settings checks passed.\n')
