library(BATTS)
# Per-tree subsampling in boots() (subsample_fraction, as ada's bag.frac).
draw <- BATTS:::.draw_subsamples
set.seed(1)
g <- rep(0:1, c(37, 120)); n <- length(g)
tr <- rep(1, n); tr[sample(n, 30)] <- 0
D <- draw(g, tr, 50, 0.5)
stopifnot(is.integer(D), nrow(D) == ceiling(sum(tr) / 2), ncol(D) == 50,
          all(D >= 0), all(D < n), all(tr[D + 1] == 1),
          all(apply(D, 2, function(v) !anyDuplicated(v))),
          all(apply(D, 2, function(v) any(g[v + 1] == 0) && any(g[v + 1] == 1))))
rng <- .Random.seed
stopifnot(identical(dim(draw(g, tr, 5, 1)), c(0L, 0L)), identical(rng, .Random.seed))

x <- matrix(runif(40), ncol = 2); gg <- rep(0:1, each = 10)
for (bad in list(0, -0.5, 1.2, NA_real_, Inf, c(.5, .5), "0.5", TRUE, matrix(.5))) {
  rng <- .Random.seed
  msg <- tryCatch({boots(x, gg, num_trees_max = 2, quiet = TRUE,
                         subsample_fraction = bad); NA_character_},
                  error = conditionMessage)
  stopifnot(grepl("subsample_fraction must", msg, fixed = TRUE), identical(rng, .Random.seed))
}

# Pure-R reference for root-only trees: leaf values use the subsample and its
# group sizes; forward-stagewise normalization uses the whole training set.
ref <- function(g, draws, lr, gradient) {
  w <- rep(1, length(g))
  for (t in seq_len(ncol(draws))) {
    S <- draws[, t] + 1
    b <- ((sum(1 / w[S][g[S] == 0]) / sum(g[S] == 0)) /
            (sum(w[S][g[S] == 1]) / sum(g[S] == 1)))^lr
    w <- w * sqrt(b)
    if (!gradient) w <- w * sqrt((sum(1 / w[g == 0]) / sum(g == 0)) /
                                   (sum(w[g == 1]) / sum(g == 1)))
  }
  w
}
set.seed(31)
xr <- cbind(runif(115), runif(115)); gr <- rep(0:1, c(45, 70))
for (gradient in c(FALSE, TRUE)) {
  set.seed(500)
  fit <- boots(xr, gr, num_trees_max = 15, max_resol = 0, learn_rate = 0.1,
               margin_scale = -1, use_gradient = gradient, quiet = TRUE,
               subsample_fraction = 0.5)
  set.seed(500)
  w <- ref(gr, draw(gr, rep(1, 115), 15, 0.5), 0.1, gradient)
  stopifnot(isTRUE(all.equal(as.numeric(fit$balance_weight_boosting_data), w,
                             tolerance = 1e-10)))
}

x4 <- cbind(c(rnorm(30), rnorm(50, .5)), rnorm(80)); g4 <- rep(0:1, c(30, 50))
for (gradient in c(FALSE, TRUE)) {
  fit_one <- function(f) boots(x4, g4, num_trees_max = 10, K_CV = 4,
                               use_gradient = gradient, quiet = TRUE,
                               subsample_fraction = f)
  set.seed(9); f1 <- fit_one(0.5)
  set.seed(9); f2 <- fit_one(0.5)
  set.seed(9); f3 <- fit_one(1)
  p <- eval_balance_weight(f1, x4)$balancing_weight_boosting
  stopifnot(identical(f1, f2), all(is.finite(f1$loss_CV_store)),
            all(is.finite(p)), all(p > 0),
            isTRUE(all.equal(as.numeric(p), as.numeric(f1$balance_weight_boosting_data),
                             tolerance = 1e-10)),
            !identical(f1$balance_weight_boosting_data, f3$balance_weight_boosting_data))
}
cat('Subsampling draw, validation, reference and reproducibility checks passed.\n')
