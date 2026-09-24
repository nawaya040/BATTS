get_arg_value <- function(args, name, required = FALSE) {
  prefix <- paste0("--", name, "=")
  hits <- args[startsWith(args, prefix)]
  if (!length(hits)) {
    if (required) stop("Missing required argument --", name)
    return(NULL)
  }
  sub(prefix, "", hits[[1L]], fixed = TRUE)
}

metric_values <- function(estimate, truth, labels) {
  group0 <- labels == 0L
  group1 <- labels == 1L
  squared_error <- (estimate - truth)^2
  finite <- is.finite(estimate)
  mse0 <- if (all(finite[group0])) mean(squared_error[group0]) else Inf
  mse1 <- if (all(finite[group1])) mean(squared_error[group1]) else Inf
  c(
    mse_group0 = mse0,
    mse_group1 = mse1,
    mse_symmetric = (mse0 + mse1) / 2,
    nonfinite = sum(!finite),
    max_abs_finite = if (any(finite)) max(abs(estimate[finite])) else NA_real_
  )
}

submitted_cdc <- function(drt_log_ratio, labels, n0, n1) {
  score <- drt_log_ratio - log(n1 / n0)
  score0 <- score[labels == 0L]
  score1 <- score[labels == 1L]
  density0 <- stats::density(score0)
  density1 <- stats::density(score1)
  f0 <- stats::approx(
    density0$x, density0$y, xout = score, rule = 2
  )$y
  f1 <- stats::approx(
    density1$x, density1$y, xout = score, rule = 2
  )$y
  list(
    estimate = log(f0 / f1),
    zero_f0 = sum(f0 == 0),
    zero_f1 = sum(f1 == 0),
    nonfinite_f0 = sum(!is.finite(f0)),
    nonfinite_f1 = sum(!is.finite(f1)),
    bandwidth0 = density0$bw,
    bandwidth1 = density1$bw
  )
}

mean_se <- function(x) {
  if (!length(x) || any(!is.finite(x))) {
    return(c(mean = NA_real_, se = NA_real_))
  }
  c(mean = mean(x), se = stats::sd(x) / sqrt(length(x)))
}

red_value <- function(x) {
  if (is.finite(x)) {
    sprintf("\\textcolor{red}{%.3f}", x)
  } else {
    "\\textcolor{red}{undefined}"
  }
}

red_se <- function(x) {
  if (is.finite(x)) {
    sprintf("\\textcolor{red}{(%.3f)}", x)
  } else {
    "\\textcolor{red}{(--)}"
  }
}

latex_rows <- function(summary_rows) {
  list(
    drt_mean = paste(vapply(summary_rows$drt_mean, red_value, character(1L)), collapse = " &"),
    drt_se = paste(vapply(summary_rows$drt_se, red_se, character(1L)), collapse = " &"),
    cdc_mean = paste(vapply(summary_rows$cdc_mean, red_value, character(1L)), collapse = " &"),
    cdc_se = paste(vapply(summary_rows$cdc_se, red_se, character(1L)), collapse = " &")
  )
}

command_args <- commandArgs(trailingOnly = FALSE)
script_arg <- command_args[startsWith(command_args, "--file=")]
script_path <- if (length(script_arg)) {
  normalizePath(sub("^--file=", "", script_arg[[1L]]), winslash = "/", mustWork = TRUE)
} else {
  NA_character_
}
args <- commandArgs(trailingOnly = TRUE)
raw_root <- normalizePath(
  get_arg_value(args, "raw-root", required = TRUE),
  winslash = "/", mustWork = TRUE
)
checksums_path <- normalizePath(
  get_arg_value(args, "checksums", required = TRUE),
  winslash = "/", mustWork = TRUE
)
output_dir <- get_arg_value(args, "output-dir", required = TRUE)

output_files <- c(
  "input_manifest.csv",
  "per_seed_metrics.csv",
  "cell_summary.csv",
  "table_1_2_adaboost_updated.tex",
  "run_metadata.txt"
)
if (dir.exists(output_dir)) {
  existing <- file.path(output_dir, output_files)
  if (any(file.exists(existing))) {
    stop("Refusing to overwrite existing outputs in ", output_dir)
  }
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_dir <- normalizePath(output_dir, winslash = "/", mustWork = TRUE)

checksums <- utils::read.csv(checksums_path, stringsAsFactors = FALSE)
required_checksum_columns <- c(
  "job_id", "output_file", "output_sha256", "output_bytes"
)
if (length(setdiff(required_checksum_columns, names(checksums)))) {
  stop("Checksum table lacks required columns")
}

specs <- data.frame(
  table_id = c(rep("table1", 6L), rep("table2", 4L)),
  cell_order = c(seq_len(6L), seq_len(4L)),
  cell_id = c(
    "2d_global_shift_balanced", "2d_global_shift_unbalanced",
    "2d_local_shift_balanced", "2d_local_shift_unbalanced",
    "2d_local_dispersion_balanced", "2d_local_dispersion_unbalanced",
    "20d_location_balanced", "20d_location_unbalanced",
    "20d_dispersion_balanced", "20d_dispersion_unbalanced"
  ),
  family = c(rep("2d", 6L), rep("20d", 4L)),
  scenario = c(
    "global_shift", "global_shift", "local_shift", "local_shift",
    "local_dispersion", "local_dispersion",
    "latent_location_shift", "latent_location_shift",
    "latent_dispersion", "latent_dispersion"
  ),
  n0 = rep(c(5000L, 9000L), 5L),
  n1 = rep(c(5000L, 1000L), 5L),
  transformed = FALSE,
  stringsAsFactors = FALSE
)

manifest_rows <- vector("list", nrow(specs) * 50L)
metric_rows <- vector("list", nrow(specs) * 50L)
row_index <- 1L

for (index_spec in seq_len(nrow(specs))) {
  spec <- specs[index_spec, , drop = FALSE]
  for (seed in seq_len(50L)) {
    job_id <- sprintf(
      "boosting_selection_%s_%s_n0-%d_n1-%d_transformed-false_seed-%03d",
      spec$family, spec$scenario, spec$n0, spec$n1, seed
    )
    expected <- checksums[checksums$job_id == job_id, , drop = FALSE]
    if (nrow(expected) != 1L) {
      stop("Expected one checksum row for ", job_id)
    }
    input_path <- file.path(raw_root, expected$output_file)
    if (!file.exists(input_path)) {
      stop("Missing input: ", input_path)
    }
    actual_bytes <- unname(file.info(input_path)$size)
    actual_sha256 <- unname(tools::sha256sum(input_path))
    if (!identical(actual_bytes, as.numeric(expected$output_bytes)) ||
        !identical(tolower(actual_sha256), tolower(expected$output_sha256))) {
      stop("Input checksum or size mismatch: ", input_path)
    }

    object <- readRDS(input_path)
    config <- object$metadata$config
    if (!identical(config$family, spec$family) ||
        !identical(config$scenario, spec$scenario) ||
        !identical(config$n0, as.integer(spec$n0)) ||
        !identical(config$n1, as.integer(spec$n1)) ||
        !identical(config$transformed, FALSE)) {
      stop("Saved configuration mismatch for ", job_id)
    }

    result <- object$result
    labels <- as.integer(result$design$labels_train)
    truth <- as.numeric(result$design$truth_train)
    drt <- as.numeric(result$adaboost$estimates$train$exponential_loss)
    if (length(labels) != spec$n0 + spec$n1 ||
        length(truth) != length(labels) || length(drt) != length(labels)) {
      stop("Saved vector lengths are inconsistent for ", job_id)
    }
    if (sum(labels == 0L) != spec$n0 || sum(labels == 1L) != spec$n1) {
      stop("Saved group sizes are inconsistent for ", job_id)
    }

    drt_metric <- metric_values(drt, truth, labels)
    stored <- result$adaboost$metrics[
      result$adaboost$metrics$selection == "exponential_loss" &
        result$adaboost$metrics$split == "train",
      , drop = FALSE
    ]
    if (nrow(stored) != 1L) {
      stop("Expected one stored DRT metric row for ", job_id)
    }
    drt_difference <- max(abs(c(
      drt_metric[["mse_group0"]] - stored$mse_group0,
      drt_metric[["mse_group1"]] - stored$mse_group1,
      drt_metric[["mse_symmetric"]] - stored$mse_symmetric
    )))
    if (!is.finite(drt_difference) || drt_difference > 1e-12) {
      stop("Recomputed DRT metric mismatch for ", job_id)
    }

    cdc_error <- NA_character_
    cdc <- tryCatch(
      submitted_cdc(drt, labels, spec$n0, spec$n1),
      error = function(error) {
        cdc_error <<- conditionMessage(error)
        NULL
      }
    )
    if (is.null(cdc)) {
      cdc_metric <- c(
        mse_group0 = NA_real_, mse_group1 = NA_real_,
        mse_symmetric = NA_real_, nonfinite = length(labels),
        max_abs_finite = NA_real_
      )
      zero_f0 <- zero_f1 <- nonfinite_f0 <- nonfinite_f1 <- NA_integer_
      bandwidth0 <- bandwidth1 <- NA_real_
    } else {
      cdc_metric <- metric_values(cdc$estimate, truth, labels)
      zero_f0 <- cdc$zero_f0
      zero_f1 <- cdc$zero_f1
      nonfinite_f0 <- cdc$nonfinite_f0
      nonfinite_f1 <- cdc$nonfinite_f1
      bandwidth0 <- cdc$bandwidth0
      bandwidth1 <- cdc$bandwidth1
    }

    selected_row <- result$adaboost$selections[
      result$adaboost$selections$criterion == "exponential_loss",
      , drop = FALSE
    ]
    if (nrow(selected_row) != 1L) {
      stop("Expected one exponential-loss selection for ", job_id)
    }

    manifest_rows[[row_index]] <- data.frame(
      table_id = spec$table_id,
      cell_order = spec$cell_order,
      cell_id = spec$cell_id,
      seed = seed,
      job_id = job_id,
      path = normalizePath(input_path, winslash = "/", mustWork = TRUE),
      bytes = actual_bytes,
      sha256 = actual_sha256,
      stringsAsFactors = FALSE
    )
    metric_rows[[row_index]] <- data.frame(
      table_id = spec$table_id,
      cell_order = spec$cell_order,
      cell_id = spec$cell_id,
      family = spec$family,
      scenario = spec$scenario,
      n0 = spec$n0,
      n1 = spec$n1,
      seed = seed,
      selected_trees = selected_row$final_selected_trees,
      drt_mse_group0 = drt_metric[["mse_group0"]],
      drt_mse_group1 = drt_metric[["mse_group1"]],
      drt_mse_symmetric = drt_metric[["mse_symmetric"]],
      drt_stored_max_abs_difference = drt_difference,
      cdc_mse_group0 = cdc_metric[["mse_group0"]],
      cdc_mse_group1 = cdc_metric[["mse_group1"]],
      cdc_mse_symmetric = cdc_metric[["mse_symmetric"]],
      cdc_nonfinite_points = cdc_metric[["nonfinite"]],
      cdc_zero_f0 = zero_f0,
      cdc_zero_f1 = zero_f1,
      cdc_nonfinite_f0 = nonfinite_f0,
      cdc_nonfinite_f1 = nonfinite_f1,
      cdc_bandwidth0 = bandwidth0,
      cdc_bandwidth1 = bandwidth1,
      cdc_error = cdc_error,
      stringsAsFactors = FALSE
    )
    row_index <- row_index + 1L
  }
}

input_manifest <- do.call(rbind, manifest_rows)
per_seed <- do.call(rbind, metric_rows)
if (nrow(per_seed) != 500L || length(unique(per_seed$cell_id)) != 10L) {
  stop("Unexpected final table dimensions")
}

summary_rows <- vector("list", nrow(specs))
for (index_spec in seq_len(nrow(specs))) {
  spec <- specs[index_spec, , drop = FALSE]
  cell <- per_seed[per_seed$cell_id == spec$cell_id, , drop = FALSE]
  if (nrow(cell) != 50L || !identical(sort(cell$seed), seq_len(50L))) {
    stop("Expected seeds 1:50 for ", spec$cell_id)
  }
  drt_summary <- mean_se(cell$drt_mse_symmetric)
  cdc_summary <- mean_se(cell$cdc_mse_symmetric)
  summary_rows[[index_spec]] <- data.frame(
    table_id = spec$table_id,
    cell_order = spec$cell_order,
    cell_id = spec$cell_id,
    n_seeds = nrow(cell),
    drt_mean = drt_summary[["mean"]],
    drt_se = drt_summary[["se"]],
    cdc_mean = cdc_summary[["mean"]],
    cdc_se = cdc_summary[["se"]],
    cdc_finite_seeds = sum(is.finite(cell$cdc_mse_symmetric)),
    cdc_failed_seeds = sum(!is.finite(cell$cdc_mse_symmetric)),
    cdc_nonfinite_points = sum(cell$cdc_nonfinite_points),
    cdc_zero_f0 = sum(cell$cdc_zero_f0, na.rm = TRUE),
    cdc_zero_f1 = sum(cell$cdc_zero_f1, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}
cell_summary <- do.call(rbind, summary_rows)
cell_summary <- cell_summary[order(cell_summary$table_id, cell_summary$cell_order), ]

table1_summary <- cell_summary[cell_summary$table_id == "table1", ]
table2_summary <- cell_summary[cell_summary$table_id == "table2", ]
table1_rows <- latex_rows(table1_summary)
table2_rows <- latex_rows(table2_summary)

table1_code <- c(
  "\\begin{table}[htb]",
  "\\centering",
  "\\caption{A comparison of the MSEs and their standard errors in the two-dimensional scenarios.}",
  "\\label{table: MSE(2D)}",
  "\\begin{tabular}{lcccccc}\\toprule",
  "&\\multicolumn{2}{c}{\\textbf{Global Shift}}&\\multicolumn{2}{c}{\\textbf{Local Shift}}&\\multicolumn{2}{c}{\\textbf{Local Dispersion}}\\\\",
  "\\cmidrule(r){2-3}\\cmidrule(r){4-5}\\cmidrule(r){6-7}",
  "$n_0 / (n_0 + n_1)$ &0.5&0.9&0.5&0.9&0.5&0.9 \\\\ \\midrule",
  "GB &0.033 &0.062 &0.035 &0.067 &0.108 &0.133 \\\\",
  "&(0.001) &(0.001) &(0.001) &(0.002) &(0.002) &(0.005) \\\\",
  "FS &0.035 &0.072 &0.035 &0.071 &0.111 &0.132 \\\\",
  "&(0.001) &(0.002) &(0.001) &(0.002) &(0.002) &(0.005) \\\\",
  "BAT &0.010 &0.018 &0.046 &0.138 &0.112 &0.190 \\\\",
  "&(0.000) &(0.001) &(0.001) &(0.002) &(0.002) &(0.006) \\\\",
  paste0("DRT (AdaBoost) &", table1_rows$drt_mean, " \\\\"),
  paste0("&", table1_rows$drt_se, " \\\\"),
  paste0("CDC (AdaBoost) &", table1_rows$cdc_mean, " \\\\"),
  paste0("&", table1_rows$cdc_se, " \\\\"),
  "KLIEP &0.156 &0.156 &0.301 &0.313 &0.304 &0.308 \\\\",
  "&(0.005) &(0.007) &(0.004) &(0.008) &(0.004) &(0.007) \\\\",
  "uLSIF &1.068 &0.422 &0.383 &0.410 &0.142 &0.211 \\\\",
  "&(0.171) &(0.096) &(0.015) &(0.022) &(0.007) &(0.010) \\\\",
  "\\bottomrule",
  "\\end{tabular}",
  "\\begin{tablenotes}",
  "\\small",
  "\\item Notes: The methods are GB (gradient boosting), FS (forward stagewise), BAT (Bayesian additive trees), DRT and CDC (density-ratio trick and calibrated discriminative classifier based on AdaBoost), and the two kernel-based methods KLIEP and uLSIF.",
  "\\item Under the submitted CDC calculation, the numbers of repetitions with nonfinite estimates were 26/50, 4/50, 29/50, and 10/50 for the balanced and unbalanced local-shift and local-dispersion cells, respectively. Unconditional 50-repetition means are therefore undefined for these cells.",
  "\\end{tablenotes}",
  "\\end{table}"
)

table2_code <- c(
  "\\begin{table}[htb]",
  "\\centering",
  "\\caption{A comparison of the MSEs and their standard errors in the 20-dimensional scenarios.}",
  "\\label{table: MSE(multi)}",
  "\\begin{tabular}{lcccc}\\toprule",
  "&\\multicolumn{2}{c}{\\textbf{Location Shift}}&\\multicolumn{2}{c}{\\textbf{Dispersion}}\\\\",
  "\\cmidrule(r){2-3}\\cmidrule(r){4-5}",
  "$n_0 / (n_0 + n_1)$ &0.5&0.9&0.5&0.9\\\\ \\midrule",
  "GB &0.073 &0.125 &0.151 &0.227 \\\\",
  "&(0.001) &(0.002) &(0.003) &(0.005) \\\\",
  "FS &0.076 &0.134 &0.158 &0.237 \\\\",
  "&(0.001) &(0.002) &(0.004) &(0.005) \\\\",
  "BAT &0.055 &0.098 &0.164 &0.338 \\\\",
  "&(0.001) &(0.002) &(0.004) &(0.008) \\\\",
  paste0("DRT (AdaBoost) &", table2_rows$drt_mean, " \\\\"),
  paste0("&", table2_rows$drt_se, " \\\\"),
  paste0("CDC (AdaBoost) &", table2_rows$cdc_mean, " \\\\"),
  paste0("&", table2_rows$cdc_se, " \\\\"),
  "KLIEP &0.614 &0.607 &0.547 &0.548 \\\\",
  "&(0.006) &(0.006) &(0.001) &(0.001) \\\\",
  "uLSIF &3.307 &3.569 &0.546 &0.547 \\\\",
  "&(0.070) &(0.094) &(0.001) &(0.001) \\\\",
  "\\bottomrule",
  "\\end{tabular}",
  "\\begin{tablenotes}",
  "\\small",
  "\\item Notes: The methods are GB (gradient boosting), FS (forward stagewise), BAT (Bayesian additive trees), DRT and CDC (density-ratio trick and calibrated discriminative classifier based on AdaBoost), and the two kernel-based methods KLIEP and uLSIF.",
  "\\end{tablenotes}",
  "\\end{table}"
)

utils::write.csv(
  input_manifest, file.path(output_dir, "input_manifest.csv"), row.names = FALSE
)
utils::write.csv(
  per_seed, file.path(output_dir, "per_seed_metrics.csv"), row.names = FALSE,
  na = ""
)
utils::write.csv(
  cell_summary, file.path(output_dir, "cell_summary.csv"), row.names = FALSE,
  na = ""
)
writeLines(
  c(
    "% Requires \\usepackage{xcolor}, \\usepackage{booktabs}, and \\usepackage{threeparttable}.",
    "% Only the DRT and CDC numerical entries are updated and colored red.",
    table1_code,
    "",
    table2_code
  ),
  file.path(output_dir, "table_1_2_adaboost_updated.tex")
)
metadata <- c(
  paste0("generated_at=", format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")),
  paste0("script=", script_path),
  paste0("raw_root=", raw_root),
  paste0("checksums=", checksums_path),
  "selection=exponential_loss",
  "cdc=submitted calculation: stats::density, approx(rule=2), log(f0/f1)",
  "stable_repair=none",
  "evaluation=symmetric training-sample MSE",
  "se=sample standard deviation of per-seed MSE divided by sqrt(50)",
  paste0("input_files=", nrow(input_manifest)),
  paste0("max_drt_metric_difference=", max(per_seed$drt_stored_max_abs_difference)),
  paste0("cdc_failed_seeds=", sum(!is.finite(per_seed$cdc_mse_symmetric))),
  paste0("R_version=", R.version.string)
)
writeLines(metadata, file.path(output_dir, "run_metadata.txt"))

cat("Created submitted-CDC Table 1/2 draft in ", output_dir, "\n", sep = "")
