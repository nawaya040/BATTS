library(BATTS)
# Flexible inputs of batts_boost()/batts_bart() and predict(): every accepted
# form must give the fit of the equivalent numeric matrix with 0/1 labels.
set.seed(51)
x <- matrix(c(rnorm(40), rnorm(40, 0.6), rnorm(80)), ncol = 2,
            dimnames = list(NULL, c("a", "b")))
g <- rep(0:1, each = 40)
ctl_b <- boost_control(num_trees_max = 8, K_CV = 2)
ctl_s <- bart_control(num_trees = 4, burnin = 5, draws = 6, keep_draws = TRUE)
core <- function(f) unclass(f)[setdiff(names(f), c("call", "control", "group_levels",
                                                    "variable_names"))]
bad <- function(expr) tryCatch({expr; NA_character_}, error = conditionMessage)
fit_b <- function(...) { set.seed(5); f <- batts_boost(..., control = ctl_b, quiet = TRUE)
                         list(fit = f, rng = .Random.seed) }
fit_s <- function(...) { set.seed(6); f <- batts_bart(..., control = ctl_s, quiet = TRUE)
                         list(fit = f, rng = .Random.seed) }
same <- function(a, b) identical(core(a$fit), core(b$fit)) && identical(a$rng, b$rng)

# 1. Reference fits; the 0/1 numeric matrix is passed through unchanged.
ref_b <- fit_b(x, g); ref_s <- fit_s(x, g)
set.seed(5); old_b <- boots(x, g, num_trees_max = 8, K_CV = 2, quiet = TRUE)
stopifnot(identical(core(ref_b$fit), unclass(old_b)),
          identical(ref_b$fit$group_levels, c("0", "1")),
          identical(ref_b$fit$variable_names, c("a", "b")))

# 2. Data frames, integer matrices and vectors.
xi <- round(x * 10); storage.mode(xi) <- "integer"
stopifnot(same(fit_b(as.data.frame(x), g), ref_b), same(fit_s(as.data.frame(x), g), ref_s),
          same(fit_b(xi, g), fit_b(xi * 1, g)),
          same(fit_b(x[, 1], g), fit_b(unname(x[, 1, drop = FALSE]), g)),
          grepl("non-numeric columns: f",
                bad(batts_boost(data.frame(a = x[, 1], f = factor(g)), g))),
          grepl("missing or infinite", bad(batts_boost(replace(x, 3, NA), g))),
          grepl("constant column 'b'", bad(batts_boost(cbind(a = x[, 1], b = 1), g))))

# 3. Group labels of other types; the first level is the numerator.
lab <- c("ctrl", "trt")[g + 1]
stopifnot(same(fit_b(x, factor(lab)), ref_b), same(fit_b(x, lab), ref_b),
          same(fit_b(x, g == 1), ref_b), same(fit_b(x, g + 1), ref_b),
          same(fit_s(x, factor(lab)), ref_s),
          identical(fit_b(x, factor(lab))$fit$group_levels, c("ctrl", "trt")),
          same(fit_b(x, factor(lab, levels = c("trt", "ctrl"))), fit_b(x, 1 - g)),
          identical(fit_b(x, factor(lab, levels = c("trt", "ctrl")))$fit$group_levels,
                    c("trt", "ctrl")),
          same(fit_b(x, factor(lab, levels = c("ctrl", "trt", "none"))), ref_b),
          grepl("exactly two", bad(batts_boost(x, rep(1:4, 20)))),
          grepl("missing values", bad(batts_boost(x, replace(g, 1, NA)))),
          grepl("one label per row", bad(batts_boost(x, g[-1]))))

# 4. Two samples.
x0 <- x[g == 0, ]; x1 <- x[g == 1, ]
two_b <- fit_b(x0, y = x1); two_s <- fit_s(x0, y = x1)
stopifnot(same(two_b, ref_b), same(two_s, ref_s),
          identical(two_b$fit$group_levels, c("x", "y")),
          same(fit_b(x0[, 1], y = x1[, 1]), fit_b(x[, 1], g)),
          grepl("exactly one of group", bad(batts_boost(x))),
          grepl("exactly one of group", bad(batts_boost(x0, g[1:40], y = x1))),
          grepl("same number of columns", bad(batts_boost(x0, y = x1[, 1]))),
          grepl("same column names", bad(batts_boost(x0, y = x1[, 2:1]))))

# 5. newdata forms; columns are matched by name when both sides have names.
fb <- ref_b$fit; fs <- ref_s$fit
nd <- x[1:5, ]
stopifnot(identical(predict(fb, nd), predict(fb, as.data.frame(nd))),
          identical(predict(fb, nd), predict(fb, nd[, 2:1])),
          identical(predict(fb, nd), predict(fb, as.data.frame(nd)[2:1])),
          identical(predict(fb, nd), predict(fb, unname(nd))),
          identical(predict(fb, nd[1, ]), predict(fb, nd[1, , drop = FALSE])),
          identical(predict(fs, nd, summary = "none"),
                    predict(fs, as.data.frame(nd)[2:1], summary = "none")),
          grepl("lacks the variable b", bad(predict(fb, data.frame(a = 1, c = 2)))),
          grepl("one point \\(2 values\\)", bad(predict(fb, 1:3))),
          grepl("2 columns", bad(predict(fb, matrix(0, 1, 3)))))
f1 <- fit_b(x[, 1], g)$fit
stopifnot(identical(predict(f1, c(-0.5, 0, 0.5)),
                    predict(f1, matrix(c(-0.5, 0, 0.5), ncol = 1))))

# 6. Points outside the fitted domain.
dom <- fb$data_info$min_max_values
far <- rbind(nd[1, ], c(dom[1, 2] + 1, 0), nd[2, ], c(0, dom[2, 1] - 1))
msg <- bad(predict(fb, far))
stopifnot(grepl("2 of 4 points", msg), grepl("variable 'a'", msg),
          grepl("variable 'b'", msg), grepl("outside = \"na\"", msg, fixed = TRUE))
p_na <- predict(fb, far, outside = "na")
stopifnot(identical(is.na(p_na), c(FALSE, TRUE, FALSE, TRUE)),
          identical(p_na[c(1, 3)], predict(fb, nd[1:2, ])),
          all(is.na(predict(fb, far[c(2, 4), ], outside = "na"))))
clamped <- pmin(pmax(far, matrix(dom[, 1], 4, 2, byrow = TRUE)),
                matrix(dom[, 2], 4, 2, byrow = TRUE))
stopifnot(identical(predict(fb, far, outside = "clamp"), predict(fb, clamped)),
          all(is.finite(predict(fb, far, outside = "clamp"))))
s_na <- predict(fs, far, outside = "na", interval = TRUE)
s_in <- predict(fs, nd[1:2, ], interval = TRUE)
stopifnot(all(is.na(s_na[c(2, 4), ])), identical(s_na[c(1, 3), ], s_in),
          identical(dim(predict(fs, far, outside = "na", summary = "none")), c(4L, 6L)),
          identical(predict(fs, far, outside = "clamp"), predict(fs, clamped)),
          identical(density_ratio(fb, far, outside = "na"),
                    predict(fb, far, type = "ratio", outside = "na")),
          grepl("should be one of", bad(predict(fb, far, outside = "drop"))))

# 7. Printing names the numerator group.
out <- capture.output(print(fit_b(x, factor(lab))$fit), print(two_s$fit),
                      print(summary(fit_b(x, factor(lab))$fit)))
stopifnot(any(grepl("p: group \"ctrl\" (numerator), q: group \"trt\"", out, fixed = TRUE)),
          any(grepl("p: group \"x\" (numerator), q: group \"y\"", out, fixed = TRUE)),
          any(grepl("group \"ctrl\": 40", out, fixed = TRUE)))

cat("Input handling checks passed.\n")
