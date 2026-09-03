#!/usr/bin/env Rscript

# Build the revision-era 20D MSE tables from completed result files only.
# No model is fitted. Legacy table cells are preserved from their saved RDS
# files, while DRT uses exponential-loss selection and CDC uses the submitted
# implementation as primary with the stable implementation kept separately.

parse_args <- function(args) {
  result <- list()
  for (arg in args) {
    if (!startsWith(arg, "--") || !grepl("=", arg, fixed = TRUE)) {
      stop("Arguments must have the form --name=value: ", arg)
    }
    pair <- strsplit(substring(arg, 3L), "=", fixed = TRUE)[[1L]]
    result[[pair[[1L]]]] <- paste(pair[-1L], collapse = "=")
  }
  result
}

required_arg <- function(args, name) {
  value <- args[[name]]
  if (is.null(value) || !nzchar(value)) {
    stop("Missing required argument --", name, "=PATH")
  }
  value
}

positive_integer <- function(value, name) {
  if (!grepl("^[0-9]+$", value)) {
    stop("--", name, " must be a positive integer")
  }
  result <- suppressWarnings(as.integer(value))
  if (is.na(result) || result < 1L) {
    stop("--", name, " must be a positive integer")
  }
  result
}

script_path <- function() {
  hit <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(hit)) return(NA_character_)
  normalizePath(sub("^--file=", "", hit[[1L]]), winslash = "/", mustWork = TRUE)
}

assert_columns <- function(data, required, label) {
  missing <- setdiff(required, names(data))
  if (length(missing)) {
    stop(label, " lacks columns: ", paste(missing, collapse = ", "))
  }
}

as_logical_strict <- function(value, label) {
  normalized <- tolower(as.character(value))
  if (any(!normalized %in% c("true", "false"))) {
    stop("Invalid logical values in ", label)
  }
  normalized == "true"
}

metric_values <- function(estimate, truth, labels) {
  finite <- is.finite(estimate)
  squared_error <- (estimate - truth)^2
  group0 <- labels == 0L
  group1 <- labels == 1L
  mse0 <- if (all(finite[group0])) mean(squared_error[group0]) else Inf
  mse1 <- if (all(finite[group1])) mean(squared_error[group1]) else Inf
  c(
    mse_group0 = mse0,
    mse_group1 = mse1,
    mse_symmetric = (mse0 + mse1) / 2,
    nonfinite_points = sum(!finite)
  )
}

submitted_cdc <- function(drt_log_ratio, labels, n0, n1) {
  score <- drt_log_ratio - log(n1 / n0)
  density0 <- stats::density(score[labels == 0L])
  density1 <- stats::density(score[labels == 1L])
  f0 <- stats::approx(density0$x, density0$y, xout = score, rule = 2)$y
  f1 <- stats::approx(density1$x, density1$y, xout = score, rule = 2)$y
  list(estimate = log(f0 / f1), f0 = f0, f1 = f1,
       bandwidth0 = density0$bw, bandwidth1 = density1$bw)
}

log_kde_at <- function(eval_points, sample_points, bandwidth) {
  if (!length(eval_points)) return(numeric())
  if (!is.finite(bandwidth) || bandwidth <= 0) {
    stop("Invalid KDE bandwidth")
  }
  vapply(eval_points, function(point) {
    log_terms <- -0.5 * ((point - sample_points) / bandwidth)^2
    maximum <- max(log_terms)
    maximum + log(mean(exp(log_terms - maximum))) - log(bandwidth) -
      0.5 * log(2 * pi)
  }, numeric(1L))
}

stable_cdc <- function(drt_log_ratio, labels, n0, n1) {
  score <- drt_log_ratio - log(n1 / n0)
  score0 <- score[labels == 0L]
  score1 <- score[labels == 1L]
  density0 <- stats::density(score0)
  density1 <- stats::density(score1)
  f0 <- stats::approx(density0$x, density0$y, xout = score, rule = 2)$y
  f1 <- stats::approx(density1$x, density1$y, xout = score, rule = 2)$y
  estimate <- log(f0) - log(f1)
  repair <- which(!is.finite(estimate))
  if (length(repair)) {
    estimate[repair] <-
      log_kde_at(score[repair], score0, density0$bw) -
      log_kde_at(score[repair], score1, density1$bw)
  }
  list(estimate = estimate, repaired_points = length(repair))
}

metric_row <- function(method, source, selection, scenario, n0, n1,
                       transformed, seed, mse_group0, mse_group1,
                       nonfinite_points = 0L) {
  data.frame(
    method = method,
    source = source,
    selection = selection,
    scenario = scenario,
    n0 = as.integer(n0),
    n1 = as.integer(n1),
    representation = if (isTRUE(transformed)) "transformed" else "raw",
    seed = as.integer(seed),
    mse_group0 = as.numeric(mse_group0),
    mse_group1 = as.numeric(mse_group1),
    mse_symmetric = (as.numeric(mse_group0) + as.numeric(mse_group1)) / 2,
    nonfinite_points = as.integer(nonfinite_points),
    stringsAsFactors = FALSE
  )
}

input_manifest_row <- function(path, source, scenario = NA_character_,
                               n0 = NA_integer_, n1 = NA_integer_,
                               transformed = NA, seed = NA_integer_) {
  data.frame(
    source = source,
    scenario = scenario,
    n0 = n0,
    n1 = n1,
    representation = if (is.na(transformed)) {
      "multiple"
    } else if (isTRUE(transformed)) {
      "transformed"
    } else {
      "raw"
    },
    seed = seed,
    path = normalizePath(path, winslash = "/", mustWork = TRUE),
    bytes = unname(file.info(path)$size),
    sha256 = unname(tools::sha256sum(path)),
    stringsAsFactors = FALSE
  )
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
legacy_root <- normalizePath(
  required_arg(args, "legacy-root"), winslash = "/", mustWork = TRUE
)
boosting_root <- normalizePath(
  required_arg(args, "boosting-root"), winslash = "/", mustWork = TRUE
)
boosting_checksums_path <- normalizePath(
  required_arg(args, "boosting-checksums"), winslash = "/", mustWork = TRUE
)
bart_raw_path <- normalizePath(
  required_arg(args, "bart-raw-seed-metrics"), winslash = "/", mustWork = TRUE
)
bart_transformed_path <- normalizePath(
  required_arg(args, "bart-transformed-seed-metrics"),
  winslash = "/", mustWork = TRUE
)
new_comparator_path <- normalizePath(
  required_arg(args, "new-comparator-csv"), winslash = "/", mustWork = TRUE
)
output_dir_arg <- required_arg(args, "output-dir")
expected_seeds <- positive_integer(
  if (is.null(args[["expected-seeds"]])) "50" else args[["expected-seeds"]],
  "expected-seeds"
)
legacy_lambda <- positive_integer(
  if (is.null(args[["legacy-lambda"]])) "5" else args[["legacy-lambda"]],
  "legacy-lambda"
)

if (dir.exists(output_dir_arg) &&
    length(list.files(output_dir_arg, all.files = TRUE, no.. = TRUE))) {
  stop("Refusing to overwrite a non-empty summary directory: ", output_dir_arg)
}

specs <- expand.grid(
  scenario = c(
    "latent_location_shift", "latent_dispersion", "global_shift", "null"
  ),
  sampling = c("balanced", "unbalanced"),
  transformed = c(FALSE, TRUE),
  stringsAsFactors = FALSE
)
specs$n0 <- ifelse(specs$sampling == "balanced", 5000L, 9000L)
specs$n1 <- ifelse(specs$sampling == "balanced", 5000L, 1000L)
scenario_order <- c(
  latent_location_shift = 1L, latent_dispersion = 2L,
  global_shift = 3L, null = 4L
)
specs <- specs[order(
  specs$transformed, scenario_order[specs$scenario],
  match(specs$sampling, c("balanced", "unbalanced"))
), ]
rownames(specs) <- NULL

primary_rows <- list()
sensitivity_rows <- list()
manifest_rows <- list()
primary_index <- 0L
sensitivity_index <- 0L
manifest_index <- 0L

# Preserve GB, FS, BAT, KLIEP, and uLSIF for the two existing scenarios from
# the exact legacy files underlying the current 20D manuscript tables.
legacy_methods <- c("GB", "FS", "BAT", "DRT_legacy_ignored", "KLIEP", "uLSIF")
for (scenario in c("latent_location_shift", "latent_dispersion")) {
  for (transformed in c(FALSE, TRUE)) {
    for (sizes in list(c(5000L, 5000L), c(9000L, 1000L))) {
      for (seed in seq_len(expected_seeds)) {
        stem_parts <- c(
          scenario, sizes,
          if (transformed) "transformed" else NULL,
          legacy_lambda, seed
        )
        path <- file.path(
          legacy_root, scenario, paste0(paste(stem_parts, collapse = "_"), ".rds")
        )
        if (!file.exists(path)) stop("Missing legacy input: ", path)
        object <- readRDS(path)
        required_fields <- c("error_group0_vec", "error_group1_vec")
        if (!all(required_fields %in% names(object)) ||
            length(object$error_group0_vec) != 6L ||
            length(object$error_group1_vec) != 6L) {
          stop("Unexpected legacy MSE schema: ", path)
        }
        for (column in c(1L, 2L, 3L, 5L, 6L)) {
          primary_index <- primary_index + 1L
          primary_rows[[primary_index]] <- metric_row(
            method = legacy_methods[[column]],
            source = "legacy_20d_rds",
            selection = "submitted_table_lineage",
            scenario = scenario,
            n0 = sizes[[1L]], n1 = sizes[[2L]], transformed = transformed,
            seed = seed,
            mse_group0 = object$error_group0_vec[[column]],
            mse_group1 = object$error_group1_vec[[column]]
          )
        }
        manifest_index <- manifest_index + 1L
        manifest_rows[[manifest_index]] <- input_manifest_row(
          path, "legacy_20d_rds", scenario, sizes[[1L]], sizes[[2L]],
          transformed, seed
        )
      }
    }
  }
}

# Reconstruct exponential-loss DRT and both CDC variants for the existing
# scenarios from the compact, checksummed boosting-selection outputs.
boosting_checksums <- utils::read.csv(
  boosting_checksums_path, stringsAsFactors = FALSE
)
assert_columns(
  boosting_checksums,
  c("job_id", "output_file", "output_sha256", "output_bytes"),
  "Boosting checksum table"
)
manifest_index <- manifest_index + 1L
manifest_rows[[manifest_index]] <- input_manifest_row(
  boosting_checksums_path, "boosting_checksum_index"
)

for (scenario in c("latent_location_shift", "latent_dispersion")) {
  for (transformed in c(FALSE, TRUE)) {
    for (sizes in list(c(5000L, 5000L), c(9000L, 1000L))) {
      for (seed in seq_len(expected_seeds)) {
        job_id <- sprintf(
          "boosting_selection_20d_%s_n0-%d_n1-%d_transformed-%s_seed-%03d",
          scenario, sizes[[1L]], sizes[[2L]],
          tolower(as.character(transformed)), seed
        )
        expected <- boosting_checksums[
          boosting_checksums$job_id == job_id, , drop = FALSE
        ]
        if (nrow(expected) != 1L) {
          stop("Expected one boosting checksum row for ", job_id)
        }
        path <- file.path(boosting_root, expected$output_file[[1L]])
        if (!file.exists(path)) stop("Missing boosting input: ", path)
        actual_bytes <- unname(file.info(path)$size)
        actual_sha256 <- unname(tools::sha256sum(path))
        if (!identical(actual_bytes, as.numeric(expected$output_bytes[[1L]])) ||
            !identical(
              tolower(actual_sha256), tolower(expected$output_sha256[[1L]])
            )) {
          stop("Boosting input size or SHA-256 mismatch: ", path)
        }
        object <- readRDS(path)
        config <- object$metadata$config
        if (!identical(config$family, "20d") ||
            !identical(config$scenario, scenario) ||
            !identical(config$n0, sizes[[1L]]) ||
            !identical(config$n1, sizes[[2L]]) ||
            !identical(config$transformed, transformed)) {
          stop("Boosting configuration mismatch: ", path)
        }
        result <- object$result
        labels <- as.integer(result$design$labels_train)
        truth <- as.numeric(result$design$truth_train)
        drt <- as.numeric(result$adaboost$estimates$train$exponential_loss)
        if (length(labels) != sum(sizes) || length(truth) != length(labels) ||
            length(drt) != length(labels) || any(!is.finite(c(truth, drt))) ||
            sum(labels == 0L) != sizes[[1L]] ||
            sum(labels == 1L) != sizes[[2L]]) {
          stop("Invalid boosting vectors: ", path)
        }
        drt_metric <- metric_values(drt, truth, labels)
        stored <- result$adaboost$metrics[
          result$adaboost$metrics$selection == "exponential_loss" &
            result$adaboost$metrics$split == "train", , drop = FALSE
        ]
        if (nrow(stored) != 1L || max(abs(c(
          drt_metric[["mse_group0"]] - stored$mse_group0,
          drt_metric[["mse_group1"]] - stored$mse_group1,
          drt_metric[["mse_symmetric"]] - stored$mse_symmetric
        ))) > 1e-12) {
          stop("Stored exponential-loss DRT metric mismatch: ", path)
        }
        primary_index <- primary_index + 1L
        primary_rows[[primary_index]] <- metric_row(
          "DRT (AdaBoost)", "boosting_selection_rds", "exponential_loss",
          scenario, sizes[[1L]], sizes[[2L]], transformed, seed,
          drt_metric[["mse_group0"]], drt_metric[["mse_group1"]],
          drt_metric[["nonfinite_points"]]
        )

        submitted <- submitted_cdc(drt, labels, sizes[[1L]], sizes[[2L]])
        submitted_metric <- metric_values(submitted$estimate, truth, labels)
        primary_index <- primary_index + 1L
        primary_rows[[primary_index]] <- metric_row(
          "CDC (AdaBoost)", "boosting_selection_rds", "submitted",
          scenario, sizes[[1L]], sizes[[2L]], transformed, seed,
          submitted_metric[["mse_group0"]], submitted_metric[["mse_group1"]],
          submitted_metric[["nonfinite_points"]]
        )

        stable <- stable_cdc(drt, labels, sizes[[1L]], sizes[[2L]])
        stable_metric <- metric_values(stable$estimate, truth, labels)
        sensitivity_index <- sensitivity_index + 1L
        sensitivity_rows[[sensitivity_index]] <- metric_row(
          "CDC (AdaBoost)", "boosting_selection_rds", "stable",
          scenario, sizes[[1L]], sizes[[2L]], transformed, seed,
          stable_metric[["mse_group0"]], stable_metric[["mse_group1"]],
          stable_metric[["nonfinite_points"]]
        )
        sensitivity_rows[[sensitivity_index]]$repaired_points <-
          stable$repaired_points

        manifest_index <- manifest_index + 1L
        manifest_rows[[manifest_index]] <- input_manifest_row(
          path, "boosting_selection_rds", scenario, sizes[[1L]], sizes[[2L]],
          transformed, seed
        )
      }
    }
  }
}

# Add GB, FS, exponential-loss DRT, submitted/stable CDC, KLIEP, and uLSIF
# for the new global-shift and null scenarios from the canonical comparator CSV.
new_comparator <- utils::read.csv(
  new_comparator_path, stringsAsFactors = FALSE, na.strings = ""
)
assert_columns(
  new_comparator,
  c(
    "method", "selection", "split", "mse_group0", "mse_group1",
    "mse_symmetric", "nonfinite", "scenario", "n0", "n1",
    "transformed", "seed"
  ),
  "New-scenario comparator CSV"
)
new_comparator$transformed <- as_logical_strict(
  new_comparator$transformed, "new comparator transformed"
)
selection_map <- data.frame(
  method = c("gb", "fs", "adaboost", "cdc", "cdc", "kliep", "ulsif"),
  selection = c(
    "native_balancing_loss", "native_balancing_loss", "exponential_loss",
    "submitted", "stable", "package_default", "package_default"
  ),
  display = c(
    "GB", "FS", "DRT (AdaBoost)", "CDC (AdaBoost)", "CDC (AdaBoost)",
    "KLIEP", "uLSIF"
  ),
  destination = c(
    "primary", "primary", "primary", "primary", "sensitivity", "primary",
    "primary"
  ),
  stringsAsFactors = FALSE
)
for (map_index in seq_len(nrow(selection_map))) {
  map <- selection_map[map_index, , drop = FALSE]
  selected <- new_comparator[
    new_comparator$method == map$method &
      new_comparator$selection == map$selection &
      new_comparator$split == "train" &
      new_comparator$scenario %in% c("global_shift", "null"), , drop = FALSE
  ]
  expected_rows <- 2L * 2L * 2L * expected_seeds
  if (nrow(selected) != expected_rows) {
    stop("Unexpected row count for new comparator ", map$method, "/", map$selection)
  }
  for (row_index in seq_len(nrow(selected))) {
    row <- selected[row_index, , drop = FALSE]
    built <- metric_row(
      map$display, "canonical_new_comparator_csv", map$selection,
      row$scenario, row$n0, row$n1, row$transformed, row$seed,
      row$mse_group0, row$mse_group1, row$nonfinite
    )
    if (!isTRUE(all.equal(
      built$mse_symmetric, row$mse_symmetric, tolerance = 1e-12
    ))) {
      stop("New comparator symmetric MSE mismatch at row ", row_index)
    }
    if (identical(map$destination, "primary")) {
      primary_index <- primary_index + 1L
      primary_rows[[primary_index]] <- built
    } else {
      sensitivity_index <- sensitivity_index + 1L
      sensitivity_rows[[sensitivity_index]] <- built
      sensitivity_rows[[sensitivity_index]]$repaired_points <- NA_integer_
    }
  }
}
manifest_index <- manifest_index + 1L
manifest_rows[[manifest_index]] <- input_manifest_row(
  new_comparator_path, "canonical_new_comparator_csv"
)

# Add BAT for both representations of the two new scenarios. The raw and
# transformed canonical summaries are separate by design.
read_bart_metrics <- function(path, transformed) {
  data <- utils::read.csv(path, stringsAsFactors = FALSE, na.strings = "")
  assert_columns(
    data, c("scenario", "n0", "n1", "seed", "group", "mse"),
    "BART seed metrics"
  )
  data <- data[
    data$scenario %in% c("global_shift", "null") &
      data$group %in% c("group0", "group1"), , drop = FALSE
  ]
  rows <- list()
  index <- 0L
  keys <- unique(data[c("scenario", "n0", "n1", "seed")])
  if (nrow(keys) != 2L * 2L * expected_seeds) {
    stop("Unexpected BART task count in ", path)
  }
  for (key_index in seq_len(nrow(keys))) {
    key <- keys[key_index, , drop = FALSE]
    cell <- data[
      data$scenario == key$scenario & data$n0 == key$n0 &
        data$n1 == key$n1 & data$seed == key$seed, , drop = FALSE
    ]
    if (nrow(cell) != 2L || !setequal(cell$group, c("group0", "group1"))) {
      stop("Expected one BART row per group for every seed in ", path)
    }
    index <- index + 1L
    rows[[index]] <- metric_row(
      "BAT", "canonical_bart_seed_metrics", "posterior_mean",
      key$scenario, key$n0, key$n1, transformed, key$seed,
      cell$mse[cell$group == "group0"], cell$mse[cell$group == "group1"]
    )
  }
  do.call(rbind, rows)
}

for (bart_spec in list(
  list(path = bart_raw_path, transformed = FALSE),
  list(path = bart_transformed_path, transformed = TRUE)
)) {
  bart_rows <- read_bart_metrics(bart_spec$path, bart_spec$transformed)
  for (row_index in seq_len(nrow(bart_rows))) {
    primary_index <- primary_index + 1L
    primary_rows[[primary_index]] <- bart_rows[row_index, , drop = FALSE]
  }
  manifest_index <- manifest_index + 1L
  manifest_rows[[manifest_index]] <- input_manifest_row(
    bart_spec$path, "canonical_bart_seed_metrics",
    transformed = bart_spec$transformed
  )
}

primary <- do.call(rbind, primary_rows)
sensitivity <- do.call(rbind, sensitivity_rows)
input_manifest <- do.call(rbind, manifest_rows)

method_order <- c(
  "GB", "FS", "BAT", "DRT (AdaBoost)", "CDC (AdaBoost)", "KLIEP", "uLSIF"
)
expected_primary_rows <- nrow(specs) * length(method_order) * expected_seeds
if (nrow(primary) != expected_primary_rows) {
  stop("Expected ", expected_primary_rows, " primary rows; found ", nrow(primary))
}
primary$key <- paste(
  primary$method, primary$scenario, primary$n0, primary$n1,
  primary$representation, primary$seed, sep = "|"
)
if (anyDuplicated(primary$key)) stop("Duplicate primary MSE identities")

summarize_metrics <- function(data, expected_per_cell) {
  key <- interaction(
    data[c("method", "scenario", "n0", "n1", "representation", "selection")],
    drop = TRUE, lex.order = TRUE
  )
  groups <- split(seq_len(nrow(data)), key)
  rows <- lapply(groups, function(indices) {
    cell <- data[indices, , drop = FALSE]
    if (nrow(cell) != expected_per_cell ||
        !identical(sort(cell$seed), seq_len(expected_per_cell))) {
      stop("Every MSE cell must contain seeds 1:", expected_per_cell)
    }
    values <- cell$mse_symmetric
    finite <- is.finite(values)
    unconditional_defined <- all(finite)
    data.frame(
      method = cell$method[[1L]],
      scenario = cell$scenario[[1L]],
      n0 = cell$n0[[1L]],
      n1 = cell$n1[[1L]],
      representation = cell$representation[[1L]],
      selection = cell$selection[[1L]],
      n_seeds = nrow(cell),
      finite_seeds = sum(finite),
      failed_seeds = sum(!finite),
      nonfinite_points = sum(cell$nonfinite_points, na.rm = TRUE),
      status = if (unconditional_defined) "defined" else "undefined_nonfinite",
      mean_mse = if (unconditional_defined) mean(values) else NA_real_,
      sd_across_seeds = if (unconditional_defined) stats::sd(values) else NA_real_,
      mcse = if (unconditional_defined) {
        stats::sd(values) / sqrt(length(values))
      } else {
        NA_real_
      },
      finite_only_mean = if (any(finite)) mean(values[finite]) else NA_real_,
      finite_only_median = if (any(finite)) stats::median(values[finite]) else NA_real_,
      finite_only_max = if (any(finite)) max(values[finite]) else NA_real_,
      stringsAsFactors = FALSE
    )
  })
  result <- do.call(rbind, rows)
  rownames(result) <- NULL
  result
}

primary_summary <- summarize_metrics(primary, expected_seeds)
sensitivity_summary <- summarize_metrics(sensitivity, expected_seeds)

make_table_wide <- function(summary, representation) {
  selected <- summary[summary$representation == representation, , drop = FALSE]
  cell_ids <- paste(
    selected$scenario,
    ifelse(selected$n0 == selected$n1, "balanced", "unbalanced"),
    sep = "_"
  )
  ordered_ids <- as.vector(rbind(
    paste0(names(scenario_order), "_balanced"),
    paste0(names(scenario_order), "_unbalanced")
  ))
  output <- data.frame(method = method_order, stringsAsFactors = FALSE)
  for (cell_id in ordered_ids) {
    for (metric in c("mean_mse", "mcse", "status", "failed_seeds")) {
      values <- vapply(method_order, function(method) {
        hit <- selected$method == method & cell_ids == cell_id
        if (sum(hit) != 1L) stop("Missing table cell: ", representation, "/", cell_id,
                                 "/", method)
        as.character(selected[[metric]][hit])
      }, character(1L))
      column <- paste(cell_id, metric, sep = "__")
      output[[column]] <- if (metric %in% c("mean_mse", "mcse")) {
        as.numeric(values)
      } else if (identical(metric, "failed_seeds")) {
        as.integer(values)
      } else {
        values
      }
    }
  }
  output
}

table_raw <- make_table_wide(primary_summary, "raw")
table_transformed <- make_table_wide(primary_summary, "transformed")

primary <- primary[order(
  match(primary$representation, c("raw", "transformed")),
  scenario_order[primary$scenario], primary$n0, match(primary$method, method_order),
  primary$seed
), setdiff(names(primary), "key")]
primary_summary <- primary_summary[order(
  match(primary_summary$representation, c("raw", "transformed")),
  scenario_order[primary_summary$scenario], primary_summary$n0,
  match(primary_summary$method, method_order)
), ]
sensitivity_summary <- sensitivity_summary[order(
  match(sensitivity_summary$representation, c("raw", "transformed")),
  scenario_order[sensitivity_summary$scenario], sensitivity_summary$n0
), ]

dir.create(output_dir_arg, recursive = TRUE, showWarnings = FALSE)
output_dir <- normalizePath(output_dir_arg, winslash = "/", mustWork = TRUE)
utils::write.csv(
  primary, file.path(output_dir, "mse_primary_by_seed.csv"), row.names = FALSE,
  na = ""
)
utils::write.csv(
  primary_summary, file.path(output_dir, "mse_primary_summary.csv"),
  row.names = FALSE, na = ""
)
utils::write.csv(
  sensitivity, file.path(output_dir, "cdc_stable_by_seed.csv"),
  row.names = FALSE, na = ""
)
utils::write.csv(
  sensitivity_summary, file.path(output_dir, "cdc_stable_summary.csv"),
  row.names = FALSE, na = ""
)
utils::write.csv(
  table_raw, file.path(output_dir, "mse_table_20d_raw.csv"),
  row.names = FALSE, na = ""
)
utils::write.csv(
  table_transformed, file.path(output_dir, "mse_table_20d_transformed.csv"),
  row.names = FALSE, na = ""
)
utils::write.csv(
  input_manifest, file.path(output_dir, "input_manifest.csv"),
  row.names = FALSE, na = ""
)

metadata <- c(
  paste0("generated_at_utc=", format(Sys.time(), tz = "UTC", usetz = TRUE)),
  paste0("script=", script_path()),
  paste0("script_sha256=", unname(tools::sha256sum(script_path()))),
  paste0("legacy_root=", legacy_root),
  paste0("boosting_root=", boosting_root),
  paste0("boosting_checksums=", boosting_checksums_path),
  paste0("bart_raw_seed_metrics=", bart_raw_path),
  paste0("bart_transformed_seed_metrics=", bart_transformed_path),
  paste0("new_comparator_csv=", new_comparator_path),
  paste0("output_dir=", output_dir),
  paste0("seeds_per_cell=", expected_seeds),
  paste0("legacy_lambda=", legacy_lambda),
  "evaluation=symmetric training-sample MSE: (group0 MSE + group1 MSE)/2",
  "primary_methods=GB,FS,BAT,DRT (AdaBoost),CDC (AdaBoost),KLIEP,uLSIF",
  "adaboost_selection=exponential_loss only",
  "classification_error=excluded from presentation and output",
  "balancing_loss=excluded from presentation and output",
  "cdc_primary=submitted stats::density plus approx(rule=2) calculation",
  "cdc_sensitivity=stable log-space calculation with direct Gaussian log-KDE repair only where needed",
  "cdc_nonfinite_rule=if any seed is nonfinite, unconditional mean and MCSE are undefined; finite-only diagnostics remain audit fields",
  "null_presentation=null MSE remains table support; calibration plot is the main null result",
  "uncertainty=sample SD and Monte Carlo SE across seed-level symmetric MSE",
  paste0("summary_r_version=", R.version.string),
  paste0("locale=", paste(Sys.getlocale(), collapse = ";"))
)
writeLines(metadata, file.path(output_dir, "summary_metadata.txt"), useBytes = TRUE)

cat("Revision MSE summary created: ", output_dir, "\n", sep = "")
