get_arg_value <- function(args, name, default = NULL) {
  prefix <- paste0("--", name, "=")
  matches <- args[startsWith(args, prefix)]
  if (!length(matches)) {
    return(default)
  }
  sub(prefix, "", matches[[1]], fixed = TRUE)
}

sort_by_index <- function(paths) {
  ids <- as.integer(sub(".*_(\\d+)\\.rds$", "\\1", basename(paths)))
  if (anyNA(ids)) {
    stop("Coverage filenames must end in an integer index followed by .rds")
  }
  paths[order(ids)]
}

to_repeat_matrix <- function(x) {
  if (is.null(dim(x))) matrix(x, ncol = 1L) else x
}

add_range_padding <- function(values, fraction = 0.06) {
  limits <- range(values, finite = TRUE)
  if (!all(is.finite(limits))) {
    stop("Cannot derive finite plotting limits")
  }
  span <- diff(limits)
  if (span == 0) {
    span <- max(1, abs(limits[[1]]))
  }
  limits + c(-1, 1) * fraction * span
}

args <- commandArgs(trailingOnly = TRUE)
results_root <- get_arg_value(args, "results-root")
png_output <- get_arg_value(args, "png-output")
pdf_output <- get_arg_value(args, "pdf-output")
setting_preset <- get_arg_value(args, "setting-preset", "balanced-unbalanced")

if (any(vapply(list(results_root, png_output, pdf_output), is.null, logical(1)))) {
  stop("Required arguments: --results-root, --png-output, and --pdf-output")
}

results_root <- normalizePath(results_root, mustWork = TRUE)
config_path <- file.path(results_root, "config.dput")
coverage_dir <- file.path(results_root, "coverage")
illustration_dir <- file.path(results_root, "illustration")

if (!file.exists(config_path) || !dir.exists(coverage_dir) || !dir.exists(illustration_dir)) {
  stop("The results root does not contain config.dput, coverage, and illustration inputs")
}

dir.create(dirname(png_output), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(pdf_output), recursive = TRUE, showWarnings = FALSE)
png_output <- normalizePath(png_output, mustWork = FALSE)
pdf_output <- normalizePath(pdf_output, mustWork = FALSE)

config <- dget(config_path)
setting_presets <- list(
  "balanced-unbalanced" = list(
    settings = list(c(2500L, 2500L), c(500L, 4500L)),
    titles = c(
      "Balanced:  n0 = 2500, n1 = 2500",
      "Unbalanced:  n0 = 500, n1 = 4500"
    )
  ),
  "balanced-sample-size" = list(
    settings = list(c(500L, 500L), c(2500L, 2500L)),
    titles = c(
      "Balanced:  n0 = n1 = 500",
      "Balanced:  n0 = n1 = 2500"
    )
  ),
  "balanced-unbalanced-supplement" = list(
    settings = list(c(500L, 500L), c(100L, 900L)),
    titles = c(
      "Balanced:  n0 = 500, n1 = 500",
      "Unbalanced:  n0 = 100, n1 = 900"
    )
  ),
  "unbalanced-sample-size" = list(
    settings = list(c(100L, 900L), c(500L, 4500L)),
    titles = c(
      "Unbalanced:  n0 = 100, n1 = 900",
      "Unbalanced:  n0 = 500, n1 = 4500"
    )
  )
)
if (!setting_preset %in% names(setting_presets)) {
  stop(
    "Unknown --setting-preset. Choose one of: ",
    paste(names(setting_presets), collapse = ", ")
  )
}
preset <- setting_presets[[setting_preset]]
required_settings <- preset$settings
setting_keys <- vapply(config$settings, paste, collapse = ":", character(1))
required_keys <- vapply(required_settings, paste, collapse = ":", character(1))
setting_indices <- match(required_keys, setting_keys)
if (anyNA(setting_indices)) {
  stop("Required Figure 1 settings are absent from config.dput")
}

coverage_files <- sort_by_index(list.files(
  coverage_dir,
  pattern = "^coverage_[0-9]+\\.rds$",
  full.names = TRUE
))
if (length(coverage_files) != config$n_repeat) {
  stop("Expected ", config$n_repeat, " coverage files; found ", length(coverage_files))
}

n_probs <- length(config$quant_probs)
n_coverage_probs <- floor(n_probs / 2)
c_probs <- 1 - config$quant_probs[seq_len(n_coverage_probs)] * 2
cov_rate_store <- array(
  NA_real_,
  dim = c(length(config$settings), n_coverage_probs, length(coverage_files))
)
for (index_repeat in seq_along(coverage_files)) {
  value <- readRDS(coverage_files[[index_repeat]])
  if (!identical(dim(value), dim(cov_rate_store[, , index_repeat]))) {
    stop("Unexpected coverage dimensions in ", coverage_files[[index_repeat]])
  }
  cov_rate_store[, , index_repeat] <- value
}
if (any(!is.finite(cov_rate_store))) {
  stop("Coverage inputs contain nonfinite values")
}

illustrations <- lapply(required_settings, function(setting) {
  path <- file.path(
    illustration_dir,
    sprintf("illustration_%d_%d.rds", setting[[1]], setting[[2]])
  )
  if (!file.exists(path)) {
    stop("Missing illustration file: ", path)
  }
  readRDS(path)
})

required_fields <- c("log_ratio_grid_true", "log_ratio_grid_post_mean", "CIs", "tau_inv_result")
for (index_setting in seq_along(illustrations)) {
  missing_fields <- setdiff(required_fields, names(illustrations[[index_setting]]))
  if (length(missing_fields)) {
    stop("Illustration input is missing fields: ", paste(missing_fields, collapse = ", "))
  }
  if (!all(c("2.5%", "97.5%") %in% rownames(illustrations[[index_setting]]$CIs))) {
    stop("Illustration credible interval rows are incomplete")
  }
}

bd_true <- 0.25 * (config$mu0 - config$mu1)^2 / (config$sig20 + config$sig21) +
  0.5 * log((config$sig20 + config$sig21) /
    (2 * sqrt(config$sig20 * config$sig21)))
bc_true <- exp(-bd_true)

style <- list(
  posterior = "#111111",
  interval = "#2C7FB8",
  truth = "#7A7A7A",
  simulation = "#C7C7C7",
  simulation_legend = "#A6A6A6",
  reference = "#555555"
)

scenario_titles <- preset$titles
panel_labels <- matrix(
  c(
    "(a) Log-density ratio", "(b) Inverse temperature", "(c) Calibration",
    "(d) Log-density ratio", "(e) Inverse temperature", "(f) Calibration"
  ),
  nrow = 2,
  byrow = TRUE
)

draw_title <- function(label) {
  old_mar <- graphics::par("mar")
  on.exit(graphics::par(mar = old_mar), add = TRUE)
  graphics::par(mar = c(0, 0, 0, 0))
  graphics::plot.new()
  graphics::text(0.5, 0.30, label, family = "sans", font = 2, cex = 1.12)
}

draw_panel_heading <- function(label) {
  graphics::title(main = label, adj = 0, line = 0.60, cex.main = 1.05, font.main = 2)
}

draw_ratio_panel <- function(result_illustration, grid_points, label, show_legend) {
  y_values <- c(
    result_illustration$log_ratio_grid_true,
    result_illustration$log_ratio_grid_post_mean,
    result_illustration$CIs["2.5%", ],
    result_illustration$CIs["97.5%", ]
  )
  graphics::plot(
    grid_points,
    result_illustration$log_ratio_grid_true,
    type = "l",
    lwd = 2.2,
    lty = 3,
    col = style$truth,
    xlab = "x",
    ylab = "Log-density ratio",
    ylim = add_range_padding(y_values)
  )
  graphics::lines(
    grid_points,
    result_illustration$log_ratio_grid_post_mean,
    lwd = 2.5,
    col = style$posterior
  )
  graphics::lines(
    grid_points,
    result_illustration$CIs["2.5%", ],
    lwd = 2.0,
    lty = 2,
    col = style$interval
  )
  graphics::lines(
    grid_points,
    result_illustration$CIs["97.5%", ],
    lwd = 2.0,
    lty = 2,
    col = style$interval
  )
  draw_panel_heading(label)
  if (show_legend) {
    graphics::legend(
      "bottomleft",
      legend = c("Posterior mean", "95% credible interval", "Truth"),
      col = c(style$posterior, style$interval, style$truth),
      lty = c(1, 2, 3),
      lwd = c(2.5, 2.0, 2.2),
      bty = "n",
      cex = 0.88,
      inset = 0.01
    )
  }
}

draw_temperature_panel <- function(result_illustration, label, show_legend) {
  density_obj <- stats::density(
    result_illustration$tau_inv_result,
    from = 0.7,
    to = 1.2,
    n = 512
  )
  graphics::plot(
    density_obj$x,
    density_obj$y,
    type = "l",
    lwd = 2.5,
    col = style$posterior,
    xlab = expression(tau^{-1}),
    ylab = "Posterior density",
    xlim = c(0.7, 1.2),
    ylim = c(0, max(density_obj$y, na.rm = TRUE) * 1.08)
  )
  graphics::abline(v = bc_true, col = style$truth, lty = 2, lwd = 2.0)
  draw_panel_heading(label)
  if (show_legend) {
    graphics::legend(
      "topright",
      legend = c("Posterior", expression("True " * tau^{-1})),
      col = c(style$posterior, style$truth),
      lty = c(1, 2),
      lwd = c(2.5, 2.0),
      bty = "n",
      cex = 0.88,
      inset = 0.01
    )
  }
}

draw_coverage_panel <- function(index_setting, label, show_legend) {
  cov_matrix <- to_repeat_matrix(cov_rate_store[index_setting, , ])
  graphics::matplot(
    c_probs,
    cov_matrix,
    type = "l",
    lty = 1,
    lwd = 0.9,
    col = rep(style$simulation, ncol(cov_matrix)),
    xlab = "Posterior uncertainty",
    ylab = "Empirical coverage",
    xlim = c(0, 1),
    ylim = c(-0.025, 1.025)
  )
  graphics::lines(c_probs, rowMeans(cov_matrix), lwd = 2.5, col = style$posterior)
  graphics::abline(a = 0, b = 1, lty = 2, lwd = 1.8, col = style$reference)
  draw_panel_heading(label)
  if (show_legend) {
    graphics::legend(
      "bottomright",
      legend = c("Average", "Repeats", "Nominal"),
      col = c(style$posterior, style$simulation_legend, style$reference),
      lty = c(1, 1, 2),
      lwd = c(2.5, 1.8, 1.8),
      bty = "n",
      cex = 0.88,
      inset = c(0.002, 0.002)
    )
  }
}

draw_figure <- function() {
  layout_matrix <- rbind(
    c(1, 1, 1),
    c(2, 3, 4),
    c(5, 5, 5),
    c(6, 7, 8)
  )
  graphics::layout(layout_matrix, heights = c(0.13, 1, 0.13, 1))
  old_par <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(old_par), add = TRUE)

  graphics::par(
    family = "sans",
    mar = c(3.65, 4.15, 2.35, 0.75),
    mgp = c(2.50, 0.78, 0),
    tcl = -0.25,
    las = 1,
    bty = "o",
    cex.axis = 0.96,
    cex.lab = 1.04,
    xaxs = "i",
    yaxs = "i"
  )

  for (row_index in seq_along(required_settings)) {
    draw_title(scenario_titles[[row_index]])
    result_illustration <- illustrations[[row_index]]
    grid_points <- if (!is.null(result_illustration$grid_points)) {
      result_illustration$grid_points
    } else {
      config$grid_points
    }
    if (length(grid_points) != length(result_illustration$log_ratio_grid_true)) {
      stop("Grid and illustration lengths do not agree")
    }
    show_legend <- row_index == 1L
    draw_ratio_panel(
      result_illustration,
      grid_points,
      panel_labels[row_index, 1],
      show_legend
    )
    draw_temperature_panel(
      result_illustration,
      panel_labels[row_index, 2],
      show_legend
    )
    draw_coverage_panel(
      setting_indices[[row_index]],
      panel_labels[row_index, 3],
      show_legend
    )
  }
}

png_type <- if (capabilities("cairo")) "cairo" else getOption("bitmapType")
grDevices::png(
  png_output,
  width = 3600,
  height = 2250,
  res = 300,
  pointsize = 22,
  bg = "white",
  type = png_type
)
draw_figure()
grDevices::dev.off()

grDevices::cairo_pdf(
  pdf_output,
  width = 12,
  height = 7.5,
  family = "sans",
  pointsize = 22,
  bg = "white"
)
draw_figure()
grDevices::dev.off()

cat("Created Figure 1 preview\n")
cat("  PNG: ", png_output, "\n", sep = "")
cat("  PDF: ", pdf_output, "\n", sep = "")
cat("  Setting preset: ", setting_preset, "\n", sep = "")
cat("  Settings: ", paste(required_keys, collapse = ", "), "\n", sep = "")
cat("  Coverage repeats: ", length(coverage_files), "\n", sep = "")
cat("  RNG used: none\n")
