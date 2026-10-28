library(BATTS)
# batts_boost()/batts_bart() are thin wrappers of boots()/batts().
set.seed(41)
x <- matrix(c(rnorm(40), rnorm(40, 0.6), rnorm(80)), ncol = 2); g <- rep(0:1, each = 40)
nd <- x[1:4, , drop = FALSE]
drop_new <- function(f) unclass(f)[setdiff(names(f), c("call", "control", "group_levels", "variable_names"))]
bad <- function(expr) tryCatch({expr; NA_character_}, error = conditionMessage)

# 1. Control defaults equal the defaults of the old functions.
b <- formals(boots); s <- formals(batts); bc <- boost_control(); sc <- bart_control()
stopifnot(identical(bc$num_trees_max, b$num_trees_max), identical(bc$K_CV, b$K_CV),
          identical(bc$depth, b$max_resol), identical(bc$learn_rate, b$learn_rate),
          identical(bc$n_bins, b$n_bins), identical(bc$margin_scale, b$margin_scale),
          identical(bc$subsample_fraction, b$subsample_fraction),
          identical(bc$min_obs_per_node, b$n_min_obs_per_node),
          identical(bc$use_gradient, b$use_gradient),
          identical(sc$num_trees, s$num_trees), identical(sc$burnin, s$size_burnin),
          identical(sc$draws, s$size_backfitting), identical(sc$thin, s$thin),
          identical(sc$prob_moves, eval(s$prob_moves)), identical(sc$lambda_0, s$lambda_0),
          identical(sc$omega_prior, eval(s$omega_prior_parameters)),
          identical(sc$tree_prior, eval(s$tree_priors)),
          identical(sc$margin_scale, s$margin_scale),
          identical(sc$keep_draws, s$output_BART_ensembles))

# 2. Default boosting fit is identical to boots(), including the RNG state.
set.seed(3); f_new <- batts_boost(x, g, quiet = TRUE); r_new <- .Random.seed
set.seed(3); f_old <- boots(x, g, quiet = TRUE);       r_old <- .Random.seed
stopifnot(identical(drop_new(f_new), unclass(f_old)), identical(r_new, r_old),
          identical(class(f_new), class(f_old)))

# 3. Every boost_control() setting maps to the matching boots() argument.
small <- list(num_trees_max = 8, K_CV = 2)
cases_b <- list(
  list(new = list(num_trees_max = 12), old = list(num_trees_max = 12)),
  list(new = list(K_CV = 3),           old = list(K_CV = 3)),
  list(new = list(depth = 2),          old = list(max_resol = 2)),
  list(new = list(learn_rate = 0.05),  old = list(learn_rate = 0.05)),
  list(new = list(n_bins = 16),        old = list(n_bins = 16)),
  list(new = list(margin_scale = 0.2), old = list(margin_scale = 0.2)),
  list(new = list(subsample_fraction = 1), old = list(subsample_fraction = 1)),
  list(new = list(min_obs_per_node = 3),   old = list(n_min_obs_per_node = 3)),
  list(new = list(use_gradient = FALSE),   old = list(use_gradient = FALSE)))
for(cs in cases_b){
  ctl <- do.call(boost_control, modifyList(small, cs$new))
  set.seed(5); a <- batts_boost(x, g, control = ctl, quiet = TRUE)
  set.seed(5); o <- do.call(boots, c(list(x, g, quiet = TRUE),
                                     modifyList(small, cs$old)))
  if(!identical(drop_new(a), unclass(o))) stop("boost_control mapping failed: ", names(cs$new))
}

# 4. Every bart_control() setting maps to the matching batts() argument.
tiny <- list(num_trees = 4, burnin = 5, draws = 6)
tiny_old <- list(num_trees = 4, size_burnin = 5, size_backfitting = 6)
cases_s <- list(
  list(new = list(),                         old = list()),
  list(new = list(num_trees = 5),            old = list(num_trees = 5)),
  list(new = list(burnin = 7),               old = list(size_burnin = 7)),
  list(new = list(burnin = NULL),            old = list(size_burnin = NULL)),
  list(new = list(draws = 9),                old = list(size_backfitting = 9)),
  list(new = list(thin = 2),                 old = list(thin = 2)),
  list(new = list(prob_moves = c(.5, .25, .25)), old = list(prob_moves = c(.5, .25, .25))),
  list(new = list(lambda_0 = 3),             old = list(lambda_0 = 3)),
  list(new = list(omega_prior = c(2, 1)),    old = list(omega_prior_parameters = c(2, 1))),
  list(new = list(tree_prior = c(0.9, 1.5)), old = list(tree_priors = c(0.9, 1.5))),
  list(new = list(margin_scale = 0.2),       old = list(margin_scale = 0.2)),
  list(new = list(keep_draws = TRUE),        old = list(output_BART_ensembles = TRUE)))
for(cs in cases_s){
  ctl <- do.call(bart_control, modifyList(tiny, cs$new, keep.null = TRUE))
  set.seed(6); a <- batts_bart(x, g, control = ctl, quiet = TRUE); ra <- .Random.seed
  set.seed(6); o <- do.call(batts, c(list(x, g, quiet = TRUE),
                                     modifyList(tiny_old, cs$old, keep.null = TRUE)))
  if(!identical(drop_new(a), unclass(o)) || !identical(ra, .Random.seed))
    stop("bart_control mapping failed: ", paste(names(cs$new), collapse = ","))
}

# 5. Invalid settings are rejected.
stopifnot(grepl("depth", bad(boost_control(depth = 0))),
          grepl("learn_rate", bad(boost_control(learn_rate = 0))),
          grepl("n_bins", bad(boost_control(n_bins = 1))),
          grepl("use_gradient", bad(boost_control(use_gradient = NA))),
          grepl("num_trees", bad(bart_control(num_trees = 0))),
          grepl("omega_prior", bad(bart_control(omega_prior = 1))),
          grepl("keep_draws", bad(bart_control(keep_draws = NA))),
          grepl("unused argument", bad(boost_control(max_resol = 2))),
          grepl("boost_control", bad(batts_boost(x, g, control = list()))),
          grepl("bart_control", bad(batts_bart(x, g, control = boost_control()))))

# 6. Without saved forests, new points are refused with a pointer to keep_draws.
set.seed(7); fs <- batts_bart(x, g, control = do.call(bart_control, tiny), quiet = TRUE)
stopifnot(grepl("keep_draws = TRUE", bad(predict(fs, nd)), fixed = TRUE),
          grepl("keep_draws = TRUE", bad(density_ratio(fs, nd)), fixed = TRUE),
          length(fitted(fs)) == nrow(x), identical(predict(fs), fitted(fs)))

# 7. The call and the settings are recorded only by the new functions.
stopifnot(is.call(f_new$call), inherits(f_new$control, "boost_control"),
          is.call(fs$call), inherits(fs$control, "bart_control"),
          is.null(f_old$call), is.null(f_old$control))

cat('batts_boost()/batts_bart() and control checks passed.\n')
