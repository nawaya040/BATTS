file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(file_arg)) {
  stop("This script must be run with Rscript.")
}
script_path <- normalizePath(
  sub("^--file=", "", file_arg[[1]]),
  winslash = "/",
  mustWork = TRUE
)

get_arg_value <- function(args, name, default = NULL) {
  prefix <- paste0("--", name, "=")
  matched <- args[startsWith(args, prefix)]
  if (!length(matched)) {
    return(default)
  }
  sub(prefix, "", matched[[1]], fixed = TRUE)
}

require_arg <- function(args, name) {
  value <- get_arg_value(args, name)
  if (is.null(value) || !nzchar(value)) {
    stop("Missing required argument --", name, "=PATH")
  }
  value
}

normalize_existing_dir <- function(path, label) {
  if (!dir.exists(path)) {
    stop(label, " does not exist: ", path)
  }
  normalizePath(path, winslash = "/", mustWork = TRUE)
}

parse_effect_ratios <- function(value) {
  ratios <- as.numeric(strsplit(value, ",", fixed = TRUE)[[1]])
  if (any(!is.finite(ratios)) ||
      any(ratios <= 1) ||
      any(diff(ratios) <= 0)) {
    stop("--effect-ratios must be strictly increasing numbers greater than 1.")
  }
  ratios
}

format_ratio <- function(value) {
  formatC(value, format = "f", digits = 2)
}

discover_detail_repeats <- function(
    results_root,
    scenario,
    n0,
    n1,
    lambda_0) {
  scenario_dir <- file.path(results_root, scenario)
  prefix <- paste(scenario, n0, n1, lambda_0, sep = "_")
  pattern <- paste0("^", prefix, "_([0-9]+)_details[.]rds$")
  file_names <- list.files(scenario_dir, pattern = pattern)
  if (!length(file_names)) {
    stop("No detail RDS files found for ", prefix)
  }
  as.integer(sub(pattern, "\\1", file_names))
}

validate_details <- function(object, n_observations, path) {
  required_names <- c(
    "log_ratio_BART_data_mean",
    "log_ratio_BART_data_quantiles"
  )
  missing_names <- setdiff(required_names, names(object))
  if (length(missing_names)) {
    stop(
      "Missing fields in ", path, ": ",
      paste(missing_names, collapse = ", ")
    )
  }

  posterior_mean <- object$log_ratio_BART_data_mean
  quantiles <- object$log_ratio_BART_data_quantiles

  if (!is.numeric(posterior_mean) ||
      length(posterior_mean) != n_observations ||
      any(!is.finite(posterior_mean))) {
    stop("Invalid log_ratio_BART_data_mean in ", path)
  }
  if (!is.matrix(quantiles) ||
      !identical(dim(quantiles), c(7L, n_observations)) ||
      any(!is.finite(quantiles))) {
    stop("Invalid log_ratio_BART_data_quantiles in ", path)
  }
  if (any(quantiles[-1L, , drop = FALSE] <
          quantiles[-nrow(quantiles), , drop = FALSE])) {
    stop("Posterior quantiles are not ordered in ", path)
  }

  list(
    posterior_mean = posterior_mean,
    quantiles = quantiles
  )
}

validate_true_log_ratio <- function(object, n_observations, path) {
  if (!is.numeric(object) ||
      length(object) != n_observations ||
      any(!is.finite(object))) {
    stop("Invalid true-log-ratio vector in ", path)
  }
  object
}

compare_saved_coverage <- function(
    object,
    true_log_ratio,
    lower,
    upper,
    weights,
    path) {
  required_names <- c("log_w_true_data", "covered_95")
  missing_names <- setdiff(required_names, names(object))
  if (length(missing_names)) {
    stop(
      "Missing fields in ", path, ": ",
      paste(missing_names, collapse = ", ")
    )
  }
  if (!identical(object$log_w_true_data, true_log_ratio)) {
    stop("True log-density ratio differs between saved files: ", path)
  }

  recomputed <- as.integer(
    lower < true_log_ratio & true_log_ratio < upper
  )
  stored <- as.integer(object$covered_95)
  if (length(stored) != length(recomputed) ||
      any(!stored %in% c(0L, 1L))) {
    stop("Invalid saved covered_95 in ", path)
  }

  data.frame(
    n_indicators_different = sum(stored != recomputed),
    indicator_agreement_rate = mean(stored == recomputed),
    stored_coverage_rate_unweighted = mean(stored),
    details_coverage_rate_unweighted = mean(recomputed),
    stored_coverage_rate_equal_mixture = sum(weights * stored),
    details_coverage_rate_equal_mixture = sum(weights * recomputed),
    stringsAsFactors = FALSE
  )
}

file_manifest_row <- function(path, setting_id, index_repeat, kind) {
  info <- file.info(path)
  data.frame(
    setting_id = setting_id,
    index_repeat = index_repeat,
    file_kind = kind,
    path = normalizePath(path, winslash = "/", mustWork = TRUE),
    bytes = unname(info$size),
    modified = format(info$mtime, "%Y-%m-%dT%H:%M:%S%z"),
    sha256 = digest::digest(
      file = path,
      algo = "sha256",
      serialize = FALSE
    ),
    stringsAsFactors = FALSE
  )
}

weighted_rate <- function(indicator, weights, indices) {
  if (!length(indices)) {
    return(NA_real_)
  }
  sum(weights[indices] * indicator[indices]) / sum(weights[indices])
}

weighted_mean_value <- function(values, weights, indices) {
  if (!length(indices)) {
    return(NA_real_)
  }
  sum(weights[indices] * values[indices]) / sum(weights[indices])
}

region_metrics <- function(
    indices,
    group,
    weights,
    abs_true_log_ratio,
    interval_width,
    any_exclusion,
    correct_detection,
    wrong_direction,
    coverage) {
  data.frame(
    n_points = length(indices),
    n_group0 = sum(group[indices] == 0L),
    n_group1 = sum(group[indices] == 1L),
    mixture_mass = sum(weights[indices]),
    mean_abs_true_log_ratio = weighted_mean_value(
      abs_true_log_ratio,
      weights,
      indices
    ),
    mean_interval_width = weighted_mean_value(
      interval_width,
      weights,
      indices
    ),
    any_exclusion_rate = weighted_rate(
      any_exclusion,
      weights,
      indices
    ),
    correct_detection_rate = weighted_rate(
      correct_detection,
      weights,
      indices
    ),
    wrong_direction_rate = weighted_rate(
      wrong_direction,
      weights,
      indices
    ),
    coverage_rate = weighted_rate(
      coverage,
      weights,
      indices
    ),
    stringsAsFactors = FALSE
  )
}

summarize_repeats <- function(data, group_columns, value_columns) {
  group_key <- interaction(
    data[group_columns],
    drop = TRUE,
    lex.order = TRUE
  )
  groups <- split(seq_len(nrow(data)), group_key)

  rows <- lapply(groups, function(indices) {
    output <- data[indices[[1]], group_columns, drop = FALSE]
    for (value_name in value_columns) {
      values <- data[[value_name]][indices]
      output[[paste0("mean_", value_name)]] <- mean(values)
      output[[paste0("median_", value_name)]] <- stats::median(values)
      output[[paste0("q10_", value_name)]] <- stats::quantile(
        values,
        probs = 0.10,
        names = FALSE,
        type = 7
      )
      output[[paste0("q90_", value_name)]] <- stats::quantile(
        values,
        probs = 0.90,
        names = FALSE,
        type = 7
      )
      output[[paste0("min_", value_name)]] <- min(values)
      output[[paste0("max_", value_name)]] <- max(values)
    }
    output$n_repeats <- length(indices)
    output
  })

  output <- do.call(rbind, rows)
  rownames(output) <- NULL
  output
}

summarize_repeats_allow_na <- function(data, group_columns, value_columns) {
  group_key <- interaction(
    data[group_columns],
    drop = TRUE,
    lex.order = TRUE
  )
  groups <- split(seq_len(nrow(data)), group_key)

  rows <- lapply(groups, function(indices) {
    output <- data[indices[[1]], group_columns, drop = FALSE]
    for (value_name in value_columns) {
      values <- data[[value_name]][indices]
      valid <- is.finite(values)
      output[[paste0("n_available_", value_name)]] <- sum(valid)
      if (!any(valid)) {
        output[[paste0("mean_", value_name)]] <- NA_real_
        output[[paste0("median_", value_name)]] <- NA_real_
        output[[paste0("q10_", value_name)]] <- NA_real_
        output[[paste0("q90_", value_name)]] <- NA_real_
        output[[paste0("min_", value_name)]] <- NA_real_
        output[[paste0("max_", value_name)]] <- NA_real_
      } else {
        values <- values[valid]
        output[[paste0("mean_", value_name)]] <- mean(values)
        output[[paste0("median_", value_name)]] <- stats::median(values)
        output[[paste0("q10_", value_name)]] <- stats::quantile(
          values,
          probs = 0.10,
          names = FALSE,
          type = 7
        )
        output[[paste0("q90_", value_name)]] <- stats::quantile(
          values,
          probs = 0.90,
          names = FALSE,
          type = 7
        )
        output[[paste0("min_", value_name)]] <- min(values)
        output[[paste0("max_", value_name)]] <- max(values)
      }
    }
    output$n_repeats <- length(indices)
    output
  })

  output <- do.call(rbind, rows)
  rownames(output) <- NULL
  output
}

plot_ribbon <- function(x, lower, upper, colour, alpha = 0.12) {
  graphics::polygon(
    c(x, rev(x)),
    c(lower, rev(upper)),
    border = NA,
    col = grDevices::adjustcolor(colour, alpha.f = alpha)
  )
}

plot_localization <- function(path, bin_summary, settings, bin_labels) {
  grDevices::png(
    path,
    width = 2600,
    height = 2200,
    res = 300,
    bg = "white"
  )
  old_par <- graphics::par(no.readonly = TRUE)
  on.exit({
    graphics::par(old_par)
    grDevices::dev.off()
  }, add = TRUE)

  graphics::par(
    mfrow = c(2, 2),
    mar = c(6.7, 4.5, 3.2, 1.2),
    oma = c(1.0, 1.5, 3.0, 0.5),
    mgp = c(2.5, 0.8, 0),
    tcl = -0.25
  )

  colours <- c(
    correct = "black",
    coverage = "#2C7FB8",
    wrong = "#D7301F"
  )
  x <- seq_along(bin_labels)

  for (setting in settings) {
    panel <- bin_summary[
      bin_summary$setting_id == setting$id,
      ,
      drop = FALSE
    ]
    panel <- panel[order(panel$bin_id), , drop = FALSE]
    if (nrow(panel) != length(bin_labels)) {
      stop("Unexpected number of effect bins for ", setting$id)
    }

    graphics::plot(
      x,
      rep(NA_real_, length(x)),
      type = "n",
      xlim = c(0.7, length(x) + 0.3),
      ylim = c(0, 1),
      xaxt = "n",
      xlab = "",
      ylab = "Rate",
      main = paste0(setting$label, " (", setting$balance, ")"),
      xaxs = "i",
      yaxs = "i",
      cex.main = 1.05
    )
    graphics::axis(
      1,
      at = x,
      labels = bin_labels,
      las = 2,
      cex.axis = 0.82
    )
    graphics::mtext(
      expression(exp(abs(g(x)))),
      side = 1,
      line = 5.0
    )
    graphics::abline(h = 0.95, lty = 3, col = "grey45")

    plot_ribbon(
      x,
      panel$q10_correct_detection_rate,
      panel$q90_correct_detection_rate,
      colours[["correct"]]
    )
    plot_ribbon(
      x,
      panel$q10_coverage_rate,
      panel$q90_coverage_rate,
      colours[["coverage"]]
    )

    graphics::lines(
      x,
      panel$mean_correct_detection_rate,
      type = "o",
      pch = 16,
      lwd = 2.2,
      col = colours[["correct"]]
    )
    graphics::lines(
      x,
      panel$mean_coverage_rate,
      type = "o",
      pch = 15,
      lwd = 2.2,
      col = colours[["coverage"]]
    )
    graphics::lines(
      x,
      panel$mean_wrong_direction_rate,
      type = "o",
      pch = 17,
      lwd = 1.7,
      lty = 2,
      col = colours[["wrong"]]
    )

    if (setting$id == settings[[1]]$id) {
      graphics::legend(
        "topleft",
        legend = c(
          "Correct-direction zero exclusion",
          "Coverage of true log ratio",
          "Wrong-direction zero exclusion",
          "Nominal 0.95"
        ),
        col = c(
          colours[["correct"]],
          colours[["coverage"]],
          colours[["wrong"]],
          "grey45"
        ),
        lty = c(1, 1, 2, 3),
        pch = c(16, 15, 17, NA),
        lwd = c(2.2, 2.2, 1.7, 1),
        bty = "n",
        cex = 0.76
      )
    }
  }

  graphics::mtext(
    "20D pointwise localization from saved 95% credible intervals",
    side = 3,
    outer = TRUE,
    line = 1.0,
    cex = 1.15
  )
}

plot_localization_logscale <- function(
    path,
    log_bin_summary,
    settings,
    log_x_max,
    max_abs_log_by_setting) {
  grDevices::png(
    path,
    width = 3200,
    height = 2200,
    res = 300,
    bg = "white"
  )
  old_par <- graphics::par(no.readonly = TRUE)
  on.exit({
    graphics::par(old_par)
    grDevices::dev.off()
  }, add = TRUE)

  graphics::layout(
    matrix(c(1, 2, 5, 3, 4, 5), nrow = 2, byrow = TRUE),
    widths = c(1, 1, 0.72)
  )
  graphics::par(
    mar = c(4.5, 4.5, 3.2, 1.2),
    oma = c(1.0, 1.5, 3.0, 0.5),
    mgp = c(2.5, 0.8, 0),
    tcl = -0.25
  )

  colours <- c(
    correct = "black",
    coverage = "#2C7FB8",
    wrong = "#D7301F",
    maximum = "#7A5195"
  )

  for (setting in settings) {
    panel <- log_bin_summary[
      log_bin_summary$setting_id == setting$id,
      ,
      drop = FALSE
    ]
    panel <- panel[order(panel$log_bin_id), , drop = FALSE]
    x <- panel$log_bin_midpoint

    graphics::plot(
      x,
      rep(NA_real_, length(x)),
      type = "n",
      xlim = c(0, log_x_max),
      ylim = c(-0.04, 1.04),
      xaxt = "n",
      xlab = "True log-density ratio (absolute value)",
      ylab = "Rate",
      main = paste0(setting$label, " (", setting$balance, ")"),
      xaxs = "i",
      yaxs = "i",
      cex.main = 1.05
    )
    graphics::axis(1, at = seq(0, log_x_max, by = 0.5))
    graphics::abline(
      v = max_abs_log_by_setting[[setting$id]],
      lty = 2,
      lwd = 1.5,
      col = colours[["maximum"]]
    )

    valid_correct <- is.finite(panel$mean_correct_detection_rate)
    valid_coverage <- is.finite(panel$mean_coverage_rate)
    if (any(valid_correct)) {
      plot_ribbon(
        x[valid_correct],
        panel$q10_correct_detection_rate[valid_correct],
        panel$q90_correct_detection_rate[valid_correct],
        colours[["correct"]]
      )
    }
    if (any(valid_coverage)) {
      plot_ribbon(
        x[valid_coverage],
        panel$q10_coverage_rate[valid_coverage],
        panel$q90_coverage_rate[valid_coverage],
        colours[["coverage"]]
      )
    }

    graphics::lines(
      x,
      panel$mean_correct_detection_rate,
      type = "l",
      lty = 1,
      lwd = 2.4,
      col = colours[["correct"]]
    )
    graphics::lines(
      x,
      panel$mean_coverage_rate,
      type = "l",
      lty = 2,
      lwd = 2.4,
      col = colours[["coverage"]]
    )
    graphics::lines(
      x,
      panel$mean_wrong_direction_rate,
      type = "l",
      lwd = 2.2,
      lty = 3,
      col = colours[["wrong"]]
    )
  }

  graphics::par(mar = c(0, 0, 0, 0))
  graphics::plot.new()
  graphics::legend(
    "center",
    legend = c(
      "Correct-direction zero exclusion",
      "Coverage of true log ratio",
      "Wrong-direction zero exclusion",
      "Maximum simulated absolute log ratio"
    ),
    col = c(
      colours[["correct"]],
      colours[["coverage"]],
      colours[["wrong"]],
      colours[["maximum"]]
    ),
    lty = c(1, 2, 3, 2),
    lwd = c(2.4, 2.4, 2.2, 1.5),
    bty = "n",
    cex = 1.05
  )

  graphics::mtext(
    "20D localization on the absolute log-density-ratio scale",
    side = 3,
    outer = TRUE,
    line = 1.0,
    cex = 1.15
  )
}

args <- commandArgs(trailingOnly = TRUE)
results_root <- normalize_existing_dir(
  require_arg(args, "results-root"),
  "Results root"
)
output_dir <- require_arg(args, "output-dir")
lambda_0 <- as.integer(get_arg_value(args, "lambda", "5"))
expected_detail_repeats <- as.integer(
  get_arg_value(args, "expected-detail-repeats", "10")
)
effect_ratios <- parse_effect_ratios(
  get_arg_value(args, "effect-ratios", "1.10,1.25,1.50,2.00")
)
log_x_max <- as.numeric(get_arg_value(args, "log-x-max", "5"))
log_bin_width <- as.numeric(get_arg_value(args, "log-bin-width", "0.25"))

if (is.na(lambda_0)) {
  stop("--lambda must be an integer.")
}
if (is.na(expected_detail_repeats) || expected_detail_repeats < 1L) {
  stop("--expected-detail-repeats must be a positive integer.")
}
if (!is.finite(log_x_max) || log_x_max <= 0) {
  stop("--log-x-max must be a positive number.")
}
if (!is.finite(log_bin_width) ||
    log_bin_width <= 0 ||
    log_bin_width > log_x_max) {
  stop("--log-bin-width must be positive and no larger than --log-x-max.")
}
if (!requireNamespace("digest", quietly = TRUE)) {
  stop("The digest package is required to record SHA-256 hashes.")
}

settings <- list(
  list(
    id = "location_balanced",
    scenario = "latent_location_shift",
    label = "Location Shift",
    n0 = 5000L,
    n1 = 5000L,
    balance = "balanced"
  ),
  list(
    id = "location_unbalanced",
    scenario = "latent_location_shift",
    label = "Location Shift",
    n0 = 9000L,
    n1 = 1000L,
    balance = "unbalanced"
  ),
  list(
    id = "dispersion_balanced",
    scenario = "latent_dispersion",
    label = "Dispersion",
    n0 = 5000L,
    n1 = 5000L,
    balance = "balanced"
  ),
  list(
    id = "dispersion_unbalanced",
    scenario = "latent_dispersion",
    label = "Dispersion",
    n0 = 9000L,
    n1 = 1000L,
    balance = "unbalanced"
  )
)

detail_repeats_by_setting <- lapply(settings, function(setting) {
  sort(discover_detail_repeats(
    results_root,
    setting$scenario,
    setting$n0,
    setting$n1,
    lambda_0
  ))
})
reference_repeats <- detail_repeats_by_setting[[1]]
if (length(reference_repeats) != expected_detail_repeats) {
  stop(
    "Expected ", expected_detail_repeats,
    " detail repeats but found ", length(reference_repeats),
    " for ", settings[[1]]$id
  )
}
for (index_setting in seq_along(settings)) {
  if (!identical(detail_repeats_by_setting[[index_setting]], reference_repeats)) {
    stop("Detail repeat IDs differ across settings.")
  }
}

effect_breaks <- c(0, log(effect_ratios), Inf)
bin_labels <- c(
  paste0("<", format_ratio(effect_ratios[[1]]), "x"),
  vapply(
    seq_len(length(effect_ratios) - 1L),
    function(index) {
      paste0(
        format_ratio(effect_ratios[[index]]),
        "-",
        format_ratio(effect_ratios[[index + 1L]]),
        "x"
      )
    },
    character(1)
  ),
  paste0(">=", format_ratio(effect_ratios[[length(effect_ratios)]]), "x")
)

log_breaks <- seq(0, log_x_max, by = log_bin_width)
if (tail(log_breaks, 1) < log_x_max) {
  log_breaks <- c(log_breaks, log_x_max)
}
log_breaks <- unique(log_breaks)
if (length(log_breaks) < 2L) {
  stop("The requested log-scale bins are invalid.")
}
log_bin_labels <- paste0(
  formatC(head(log_breaks, -1L), format = "f", digits = 2),
  "-",
  formatC(tail(log_breaks, -1L), format = "f", digits = 2)
)
log_bin_midpoints <- (
  head(log_breaks, -1L) + tail(log_breaks, -1L)
) / 2

bin_rows <- list()
log_bin_rows <- list()
threshold_rows <- list()
manifest_rows <- list()
consistency_rows <- list()
bin_index <- 0L
log_bin_index <- 0L
threshold_index <- 0L
manifest_index <- 0L
consistency_index <- 0L
max_abs_log_by_setting <- setNames(
  rep(-Inf, length(settings)),
  vapply(settings, function(setting) setting$id, character(1))
)
outside_log_range_by_setting <- setNames(
  integer(length(settings)),
  names(max_abs_log_by_setting)
)

for (setting in settings) {
  n_observations <- setting$n0 + setting$n1
  group <- c(rep(0L, setting$n0), rep(1L, setting$n1))
  mixture_weights <- c(
    rep(0.5 / setting$n0, setting$n0),
    rep(0.5 / setting$n1, setting$n1)
  )
  if (abs(sum(mixture_weights) - 1) > 1e-12) {
    stop("Mixture weights do not sum to one for ", setting$id)
  }

  for (index_repeat in reference_repeats) {
    base_name <- paste(
      setting$scenario,
      setting$n0,
      setting$n1,
      lambda_0,
      index_repeat,
      sep = "_"
    )
    scenario_dir <- file.path(results_root, setting$scenario)
    details_path <- file.path(
      scenario_dir,
      paste0(base_name, "_details.rds")
    )
    true_ratio_path <- file.path(
      scenario_dir,
      paste0(base_name, "_true_log_ratio.rds")
    )
    coverage_path <- file.path(
      scenario_dir,
      paste0(base_name, "_coverage.rds")
    )
    required_paths <- c(details_path, true_ratio_path, coverage_path)
    missing_paths <- required_paths[!file.exists(required_paths)]
    if (length(missing_paths)) {
      stop("Missing input files: ", paste(missing_paths, collapse = ", "))
    }

    details <- validate_details(
      readRDS(details_path),
      n_observations,
      details_path
    )
    true_log_ratio <- validate_true_log_ratio(
      readRDS(true_ratio_path),
      n_observations,
      true_ratio_path
    )

    lower <- details$quantiles[1L, ]
    upper <- details$quantiles[7L, ]
    coverage_comparison <- compare_saved_coverage(
      readRDS(coverage_path),
      true_log_ratio,
      lower,
      upper,
      mixture_weights,
      coverage_path
    )
    consistency_index <- consistency_index + 1L
    consistency_rows[[consistency_index]] <- cbind(
      data.frame(
        setting_id = setting$id,
        scenario = setting$label,
        balance = setting$balance,
        n0 = setting$n0,
        n1 = setting$n1,
        index_repeat = index_repeat,
        stringsAsFactors = FALSE
      ),
      coverage_comparison
    )

    abs_true_log_ratio <- abs(true_log_ratio)
    interval_width <- upper - lower
    any_exclusion <- lower > 0 | upper < 0
    correct_detection <- (
      (true_log_ratio > 0 & lower > 0) |
      (true_log_ratio < 0 & upper < 0)
    )
    wrong_direction <- (
      (true_log_ratio > 0 & upper < 0) |
      (true_log_ratio < 0 & lower > 0)
    )
    coverage <- lower < true_log_ratio & true_log_ratio < upper

    if (any(correct_detection & wrong_direction)) {
      stop("Correct and wrong-direction exclusions overlap in ", details_path)
    }
    if (!identical(any_exclusion, correct_detection | wrong_direction)) {
      stop("Zero-exclusion classification is incomplete in ", details_path)
    }

    effect_bin <- cut(
      abs_true_log_ratio,
      breaks = effect_breaks,
      labels = FALSE,
      right = FALSE,
      include.lowest = TRUE
    )
    if (anyNA(effect_bin)) {
      stop("Failed to assign effect bins in ", true_ratio_path)
    }

    max_abs_log_by_setting[[setting$id]] <- max(
      max_abs_log_by_setting[[setting$id]],
      max(abs_true_log_ratio)
    )
    in_log_range <- abs_true_log_ratio <= log_x_max
    outside_log_range_by_setting[[setting$id]] <- (
      outside_log_range_by_setting[[setting$id]] +
      sum(!in_log_range)
    )
    log_effect_bin <- rep(NA_integer_, n_observations)
    if (any(in_log_range)) {
      values_for_cut <- pmin(
        abs_true_log_ratio[in_log_range],
        log_x_max - sqrt(.Machine$double.eps) * max(1, log_x_max)
      )
      log_effect_bin[in_log_range] <- cut(
        values_for_cut,
        breaks = log_breaks,
        labels = FALSE,
        right = FALSE,
        include.lowest = TRUE
      )
    }
    if (anyNA(log_effect_bin[in_log_range])) {
      stop("Failed to assign log-scale effect bins in ", true_ratio_path)
    }

    for (bin_id in seq_along(bin_labels)) {
      indices <- which(effect_bin == bin_id)
      if (!length(indices)) {
        stop(
          "Empty effect bin ", bin_labels[[bin_id]],
          " in ", true_ratio_path
        )
      }
      bin_index <- bin_index + 1L
      bin_rows[[bin_index]] <- cbind(
        data.frame(
          setting_id = setting$id,
          scenario = setting$label,
          balance = setting$balance,
          n0 = setting$n0,
          n1 = setting$n1,
          index_repeat = index_repeat,
          bin_id = bin_id,
          effect_bin = bin_labels[[bin_id]],
          stringsAsFactors = FALSE
        ),
        region_metrics(
          indices,
          group,
          mixture_weights,
          abs_true_log_ratio,
          interval_width,
          any_exclusion,
          correct_detection,
          wrong_direction,
          coverage
        )
      )
    }

    for (log_bin_id in seq_along(log_bin_labels)) {
      indices <- which(log_effect_bin == log_bin_id)
      log_bin_index <- log_bin_index + 1L
      log_bin_rows[[log_bin_index]] <- cbind(
        data.frame(
          setting_id = setting$id,
          scenario = setting$label,
          balance = setting$balance,
          n0 = setting$n0,
          n1 = setting$n1,
          index_repeat = index_repeat,
          log_bin_id = log_bin_id,
          log_bin = log_bin_labels[[log_bin_id]],
          log_bin_lower = log_breaks[[log_bin_id]],
          log_bin_upper = log_breaks[[log_bin_id + 1L]],
          log_bin_midpoint = log_bin_midpoints[[log_bin_id]],
          stringsAsFactors = FALSE
        ),
        region_metrics(
          indices,
          group,
          mixture_weights,
          abs_true_log_ratio,
          interval_width,
          any_exclusion,
          correct_detection,
          wrong_direction,
          coverage
        )
      )
    }

    for (effect_ratio in effect_ratios) {
      delta <- log(effect_ratio)
      region_definitions <- list(
        at_least = which(abs_true_log_ratio >= delta),
        below = which(abs_true_log_ratio < delta)
      )
      for (region_type in names(region_definitions)) {
        indices <- region_definitions[[region_type]]
        if (!length(indices)) {
          stop(
            "Empty threshold region for effect ratio ",
            effect_ratio, " in ", true_ratio_path
          )
        }
        threshold_index <- threshold_index + 1L
        threshold_rows[[threshold_index]] <- cbind(
          data.frame(
            setting_id = setting$id,
            scenario = setting$label,
            balance = setting$balance,
            n0 = setting$n0,
            n1 = setting$n1,
            index_repeat = index_repeat,
            effect_ratio_threshold = effect_ratio,
            delta_log_ratio = delta,
            region_type = region_type,
            stringsAsFactors = FALSE
          ),
          region_metrics(
            indices,
            group,
            mixture_weights,
            abs_true_log_ratio,
            interval_width,
            any_exclusion,
            correct_detection,
            wrong_direction,
            coverage
          )
        )
      }
    }

    for (manifest_item in list(
      list(path = details_path, kind = "details"),
      list(path = true_ratio_path, kind = "true_log_ratio"),
      list(path = coverage_path, kind = "coverage")
    )) {
      manifest_index <- manifest_index + 1L
      manifest_rows[[manifest_index]] <- file_manifest_row(
        manifest_item$path,
        setting$id,
        index_repeat,
        manifest_item$kind
      )
    }
  }
}

bin_repeat <- do.call(rbind, bin_rows)
log_bin_repeat <- do.call(rbind, log_bin_rows)
threshold_repeat <- do.call(rbind, threshold_rows)
manifest <- do.call(rbind, manifest_rows)
input_consistency <- do.call(rbind, consistency_rows)

summary_values <- c(
  "n_points",
  "n_group0",
  "n_group1",
  "mixture_mass",
  "mean_abs_true_log_ratio",
  "mean_interval_width",
  "any_exclusion_rate",
  "correct_detection_rate",
  "wrong_direction_rate",
  "coverage_rate"
)
bin_summary <- summarize_repeats(
  bin_repeat,
  c(
    "setting_id",
    "scenario",
    "balance",
    "n0",
    "n1",
    "bin_id",
    "effect_bin"
  ),
  summary_values
)
log_bin_summary <- summarize_repeats_allow_na(
  log_bin_repeat,
  c(
    "setting_id",
    "scenario",
    "balance",
    "n0",
    "n1",
    "log_bin_id",
    "log_bin",
    "log_bin_lower",
    "log_bin_upper",
    "log_bin_midpoint"
  ),
  summary_values
)
threshold_summary <- summarize_repeats(
  threshold_repeat,
  c(
    "setting_id",
    "scenario",
    "balance",
    "n0",
    "n1",
    "effect_ratio_threshold",
    "delta_log_ratio",
    "region_type"
  ),
  summary_values
)

setting_order <- setNames(
  seq_along(settings),
  vapply(settings, function(setting) setting$id, character(1))
)
bin_summary <- bin_summary[
  order(setting_order[bin_summary$setting_id], bin_summary$bin_id),
  ,
  drop = FALSE
]
log_bin_summary <- log_bin_summary[
  order(
    setting_order[log_bin_summary$setting_id],
    log_bin_summary$log_bin_id
  ),
  ,
  drop = FALSE
]
log_bin_summary$max_abs_true_log_ratio_over_repeats <- (
  max_abs_log_by_setting[log_bin_summary$setting_id]
)
threshold_summary <- threshold_summary[
  order(
    setting_order[threshold_summary$setting_id],
    threshold_summary$effect_ratio_threshold,
    threshold_summary$region_type
  ),
  ,
  drop = FALSE
]

output_paths <- c(
  file.path(output_dir, "localization_20D_effect_bins_repeat.csv"),
  file.path(output_dir, "localization_20D_effect_bins_summary.csv"),
  file.path(output_dir, "localization_20D_log_bins_repeat.csv"),
  file.path(output_dir, "localization_20D_log_bins_summary.csv"),
  file.path(output_dir, "localization_20D_thresholds_repeat.csv"),
  file.path(output_dir, "localization_20D_thresholds_summary.csv"),
  file.path(output_dir, "localization_20D_detection.png"),
  file.path(output_dir, "localization_20D_detection_continuous_0_2p5.png"),
  file.path(output_dir, "localization_20D_input_manifest.csv"),
  file.path(output_dir, "localization_20D_input_consistency.csv"),
  file.path(output_dir, "localization_20D_run_metadata.txt")
)
existing_outputs <- output_paths[file.exists(output_paths)]
if (length(existing_outputs)) {
  stop(
    "Refusing to overwrite existing outputs: ",
    paste(existing_outputs, collapse = ", ")
  )
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_dir <- normalizePath(output_dir, winslash = "/", mustWork = TRUE)

bin_repeat_path <- file.path(
  output_dir,
  "localization_20D_effect_bins_repeat.csv"
)
bin_summary_path <- file.path(
  output_dir,
  "localization_20D_effect_bins_summary.csv"
)
log_bin_repeat_path <- file.path(
  output_dir,
  "localization_20D_log_bins_repeat.csv"
)
log_bin_summary_path <- file.path(
  output_dir,
  "localization_20D_log_bins_summary.csv"
)
threshold_repeat_path <- file.path(
  output_dir,
  "localization_20D_thresholds_repeat.csv"
)
threshold_summary_path <- file.path(
  output_dir,
  "localization_20D_thresholds_summary.csv"
)
plot_path <- file.path(output_dir, "localization_20D_detection.png")
logscale_plot_path <- file.path(
  output_dir,
  "localization_20D_detection_continuous_0_2p5.png"
)
manifest_path <- file.path(
  output_dir,
  "localization_20D_input_manifest.csv"
)
consistency_path <- file.path(
  output_dir,
  "localization_20D_input_consistency.csv"
)
metadata_path <- file.path(
  output_dir,
  "localization_20D_run_metadata.txt"
)

utils::write.csv(bin_repeat, bin_repeat_path, row.names = FALSE)
utils::write.csv(bin_summary, bin_summary_path, row.names = FALSE)
utils::write.csv(
  log_bin_repeat,
  log_bin_repeat_path,
  row.names = FALSE,
  na = ""
)
utils::write.csv(
  log_bin_summary,
  log_bin_summary_path,
  row.names = FALSE,
  na = ""
)
utils::write.csv(
  threshold_repeat,
  threshold_repeat_path,
  row.names = FALSE
)
utils::write.csv(
  threshold_summary,
  threshold_summary_path,
  row.names = FALSE
)
utils::write.csv(manifest, manifest_path, row.names = FALSE)
utils::write.csv(input_consistency, consistency_path, row.names = FALSE)
plot_localization(plot_path, bin_summary, settings, bin_labels)
plot_localization_logscale(
  logscale_plot_path,
  log_bin_summary,
  settings,
  log_x_max,
  max_abs_log_by_setting
)

metadata <- c(
  paste0("generated_at=", format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")),
  paste0("script_path=", script_path),
  paste0(
    "script_sha256=",
    digest::digest(file = script_path, algo = "sha256", serialize = FALSE)
  ),
  paste0("results_root=", results_root),
  paste0("output_dir=", output_dir),
  paste0("lambda_0=", lambda_0),
  paste0(
    "detail_repeat_ids=",
    paste(reference_repeats, collapse = ",")
  ),
  paste0("n_detail_repeats=", length(reference_repeats)),
  paste0("effect_ratio_breaks=", paste(effect_ratios, collapse = ",")),
  paste0("log_x_max=", log_x_max),
  paste0("log_bin_width=", log_bin_width),
  paste0(
    "max_abs_log_ratio_by_setting=",
    paste(
      paste0(
        names(max_abs_log_by_setting),
        ":",
        format(max_abs_log_by_setting, digits = 8)
      ),
      collapse = ","
    )
  ),
  paste0(
    "points_outside_log_range_by_setting=",
    paste(
      paste0(
        names(outside_log_range_by_setting),
        ":",
        outside_log_range_by_setting
      ),
      collapse = ","
    )
  ),
  "credible_interval=stored 0.025 and 0.975 posterior quantiles",
  "evaluation_points=all observed sample points",
  "evaluation_weighting=0.5 total weight for group 0 and 0.5 for group 1",
  "correct_detection=true log ratio positive and lower bound above zero, or true log ratio negative and upper bound below zero",
  "wrong_direction=true log ratio positive and upper bound below zero, or true log ratio negative and lower bound above zero",
  "coverage_source=coverage and zero-exclusion metrics are recomputed from the stored detail-file interval endpoints",
  "coverage_consistency_check=the separately generated coverage-file indicators are compared but are not required to equal the detail-file indicators",
  "uncertainty_summary=mean and empirical 0.10/0.90 quantiles across saved detail repeats",
  paste0("R_version=", R.version.string),
  "",
  "sessionInfo:",
  capture.output(sessionInfo())
)
writeLines(metadata, metadata_path, useBytes = TRUE)

cat("20D localization analysis created.\n")
cat("Figure: ", normalizePath(plot_path), "\n", sep = "")
cat(
  "Log-scale figure: ",
  normalizePath(logscale_plot_path),
  "\n",
  sep = ""
)
cat("Effect-bin summary: ", normalizePath(bin_summary_path), "\n", sep = "")
cat(
  "Log-bin summary: ",
  normalizePath(log_bin_summary_path),
  "\n",
  sep = ""
)
cat(
  "Threshold summary: ",
  normalizePath(threshold_summary_path),
  "\n",
  sep = ""
)
cat("Manifest: ", normalizePath(manifest_path), "\n", sep = "")
cat(
  "Input consistency: ",
  normalizePath(consistency_path),
  "\n",
  sep = ""
)
cat("Metadata: ", normalizePath(metadata_path), "\n", sep = "")
