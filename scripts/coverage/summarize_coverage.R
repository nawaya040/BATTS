#!/usr/bin/env Rscript

# Summarize the completed canonical coverage grid without refitting any model.
# Existing result files are treated as read-only and every result is checked
# against its sidecar manifest before it contributes to the summary.

parse_args <- function(args) {
  values <- list()
  for (arg in args) {
    if (!startsWith(arg, "--") || !grepl("=", arg, fixed = TRUE)) {
      stop("Arguments must have the form --name=value: ", arg)
    }
    pair <- strsplit(substring(arg, 3L), "=", fixed = TRUE)[[1L]]
    values[[pair[[1L]]]] <- paste(pair[-1L], collapse = "=")
  }
  values
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
  if (!length(hit)) {
    return(NA_character_)
  }
  normalizePath(sub("^--file=", "", hit[[1L]]), winslash = "/", mustWork = TRUE)
}

assert_columns <- function(data, required, label) {
  missing <- setdiff(required, names(data))
  if (length(missing)) {
    stop(label, " lacks columns: ", paste(missing, collapse = ", "))
  }
}

summary_statistics <- function(values) {
  if (!length(values) || any(!is.finite(values))) {
    stop("Coverage summaries require finite values")
  }
  c(
    mean = mean(values),
    sd = if (length(values) > 1L) stats::sd(values) else NA_real_,
    mcse = if (length(values) > 1L) {
      stats::sd(values) / sqrt(length(values))
    } else {
      NA_real_
    },
    min = min(values),
    max = max(values)
  )
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
input_dir <- normalizePath(
  required_arg(args, "input-dir"), winslash = "/", mustWork = TRUE
)
output_dir_arg <- required_arg(args, "output-dir")
expected_jobs <- positive_integer(
  if (is.null(args[["expected-jobs"]])) "900" else args[["expected-jobs"]],
  "expected-jobs"
)
expected_seeds <- positive_integer(
  if (is.null(args[["expected-seeds"]])) "50" else args[["expected-seeds"]],
  "expected-seeds"
)

if (dir.exists(output_dir_arg) &&
    length(list.files(output_dir_arg, all.files = TRUE, no.. = TRUE))) {
  stop("Refusing to overwrite a non-empty summary directory: ", output_dir_arg)
}

manifest_paths <- sort(list.files(
  input_dir, pattern = "[.]manifest[.]rds$", full.names = TRUE
))
result_paths <- sort(list.files(
  input_dir, pattern = "[.]rds$", full.names = TRUE
))
result_paths <- result_paths[!grepl("[.]manifest[.]rds$", result_paths)]
if (length(manifest_paths) != expected_jobs || length(result_paths) != expected_jobs) {
  stop(
    "Expected ", expected_jobs, " result/manifest pairs; found ",
    length(result_paths), " results and ", length(manifest_paths), " manifests"
  )
}

seed_rows <- vector("list", expected_jobs)
manifest_rows <- vector("list", expected_jobs)
job_keys <- character(expected_jobs)

for (index in seq_along(manifest_paths)) {
  manifest_path <- manifest_paths[[index]]
  manifest <- readRDS(manifest_path)
  required_manifest <- c(
    "schema_version", "status", "family", "job_id", "seed", "config",
    "commit_hash", "source_hashes", "batts_install_hashes", "r_version",
    "rng_kind", "parallel_backend", "core_count", "output_file",
    "output_sha256", "output_bytes"
  )
  missing_manifest <- setdiff(required_manifest, names(manifest))
  if (length(missing_manifest)) {
    stop("Manifest lacks fields: ", manifest_path, ": ",
         paste(missing_manifest, collapse = ", "))
  }
  if (!identical(manifest$status, "CANONICAL")) {
    stop("Non-canonical input: ", manifest_path)
  }
  result_path <- file.path(input_dir, manifest$output_file)
  if (!file.exists(result_path)) {
    stop("Missing result named by manifest: ", result_path)
  }
  actual_bytes <- unname(file.info(result_path)$size)
  actual_sha256 <- unname(tools::sha256sum(result_path))
  if (!identical(actual_bytes, as.numeric(manifest$output_bytes)) ||
      !identical(tolower(actual_sha256), tolower(manifest$output_sha256))) {
    stop("Result size or SHA-256 mismatch: ", result_path)
  }

  object <- readRDS(result_path)
  if (!is.list(object) || !all(c("metadata", "result") %in% names(object))) {
    stop("Unexpected result schema: ", result_path)
  }
  metadata <- object$metadata
  result <- object$result
  identity_fields <- c(
    "schema_version", "status", "family", "job_id", "seed", "config",
    "commit_hash", "source_hashes", "batts_install_hashes"
  )
  mismatch <- identity_fields[!vapply(identity_fields, function(field) {
    identical(metadata[[field]], manifest[[field]])
  }, logical(1L))]
  if (length(mismatch)) {
    stop("Result/manifest identity mismatch for ", manifest$job_id, ": ",
         paste(mismatch, collapse = ", "))
  }

  config <- metadata$config
  n0 <- as.integer(config$n0)
  n1 <- as.integer(config$n1)
  if (anyNA(c(n0, n1)) || any(c(n0, n1) < 1L)) {
    stop("Invalid group sizes for ", manifest$job_id)
  }
  required_result <- c(
    "nominal_mass", "coverage_rate_BAT", "is_included_mat",
    "lower_prob", "upper_prob", "symmetry_max_abs"
  )
  missing_result <- setdiff(required_result, names(result))
  if (length(missing_result)) {
    stop("Coverage result lacks fields for ", manifest$job_id, ": ",
         paste(missing_result, collapse = ", "))
  }
  included <- result$is_included_mat
  nominal_mass <- as.numeric(result$nominal_mass)
  if (!is.matrix(included) ||
      !identical(dim(included), c(length(nominal_mass), n0 + n1)) ||
      length(nominal_mass) != 99L) {
    stop("Unexpected coverage dimensions for ", manifest$job_id)
  }
  if (any(!included %in% c(FALSE, TRUE, 0, 1)) ||
      any(!is.finite(nominal_mass)) || any(diff(nominal_mass) >= 0)) {
    stop("Invalid coverage values or nominal grid for ", manifest$job_id)
  }
  pooled <- rowMeans(included)
  if (max(abs(pooled - result$coverage_rate_BAT)) > 1e-12) {
    stop("Stored coverage differs from rowMeans(is_included_mat): ", result_path)
  }
  if (max(abs(nominal_mass - (result$upper_prob - result$lower_prob))) > 1e-12 ||
      !is.finite(result$symmetry_max_abs) || result$symmetry_max_abs > 1e-12) {
    stop("Credible-interval probability grid is not symmetric: ", result_path)
  }

  # The canonical runners construct data as group 0 followed by group 1.
  # This is checked against n0+n1 above and recorded in the output metadata.
  group0 <- rowMeans(included[, seq_len(n0), drop = FALSE])
  group1 <- rowMeans(included[, n0 + seq_len(n1), drop = FALSE])
  equal_mixture <- (group0 + group1) / 2
  scenario <- if (is.null(config$scenario)) "one_dimensional" else config$scenario
  representation <- if (identical(metadata$family, "20d")) {
    if (isTRUE(config$transformed)) "transformed" else "raw"
  } else {
    "not_applicable"
  }
  job_key <- paste(
    metadata$family, scenario, n0, n1, representation, metadata$seed, sep = "|"
  )
  job_keys[[index]] <- job_key

  seed_rows[[index]] <- data.frame(
    family = metadata$family,
    scenario = scenario,
    n0 = n0,
    n1 = n1,
    representation = representation,
    seed = as.integer(metadata$seed),
    nominal_mass = rep(nominal_mass, times = 4L),
    group = rep(
      c("all_pooled", "group0", "group1", "equal_mixture"),
      each = length(nominal_mass)
    ),
    coverage = c(pooled, group0, group1, equal_mixture),
    stringsAsFactors = FALSE
  )
  manifest_rows[[index]] <- data.frame(
    job_id = metadata$job_id,
    family = metadata$family,
    scenario = scenario,
    n0 = n0,
    n1 = n1,
    representation = representation,
    seed = as.integer(metadata$seed),
    result_path = normalizePath(result_path, winslash = "/", mustWork = TRUE),
    result_bytes = actual_bytes,
    result_sha256 = actual_sha256,
    manifest_path = normalizePath(manifest_path, winslash = "/", mustWork = TRUE),
    manifest_sha256 = unname(tools::sha256sum(manifest_path)),
    commit_hash = metadata$commit_hash,
    r_version = metadata$r_version,
    parallel_backend = metadata$parallel_backend,
    core_count = metadata$core_count,
    stringsAsFactors = FALSE
  )
}

if (anyDuplicated(job_keys)) {
  stop("Duplicate canonical coverage job identities")
}

seed_coverage <- do.call(rbind, seed_rows)
input_manifest <- do.call(rbind, manifest_rows)
expected_seed_rows <- expected_jobs * 99L * 4L
if (nrow(seed_coverage) != expected_seed_rows) {
  stop("Unexpected seed-level row count")
}

cell_key <- interaction(
  seed_coverage[c(
    "family", "scenario", "n0", "n1", "representation", "group",
    "nominal_mass"
  )],
  drop = TRUE, lex.order = TRUE
)
groups <- split(seq_len(nrow(seed_coverage)), cell_key)
summary_rows <- lapply(groups, function(indices) {
  cell <- seed_coverage[indices, , drop = FALSE]
  if (nrow(cell) != expected_seeds ||
      !identical(sort(cell$seed), seq_len(expected_seeds))) {
    stop("Expected seeds 1:", expected_seeds, " for every coverage cell")
  }
  statistics <- summary_statistics(cell$coverage)
  data.frame(
    family = cell$family[[1L]],
    scenario = cell$scenario[[1L]],
    n0 = cell$n0[[1L]],
    n1 = cell$n1[[1L]],
    representation = cell$representation[[1L]],
    group = cell$group[[1L]],
    nominal_mass = cell$nominal_mass[[1L]],
    n_seeds = nrow(cell),
    mean_coverage = statistics[["mean"]],
    sd_across_seeds = statistics[["sd"]],
    mcse = statistics[["mcse"]],
    min_coverage = statistics[["min"]],
    max_coverage = statistics[["max"]],
    stringsAsFactors = FALSE
  )
})
coverage_summary <- do.call(rbind, summary_rows)
coverage_summary <- coverage_summary[order(
  coverage_summary$family, coverage_summary$scenario,
  coverage_summary$n0, coverage_summary$n1,
  coverage_summary$representation, coverage_summary$group,
  coverage_summary$nominal_mass
), ]
rownames(coverage_summary) <- NULL

coverage_95 <- coverage_summary[
  abs(coverage_summary$nominal_mass - 0.95) < 1e-12, , drop = FALSE
]
expected_95_rows <- expected_jobs / expected_seeds * 4L
if (nrow(coverage_95) != expected_95_rows) {
  stop("Expected exactly one 95% row per design cell and coverage group")
}

dir.create(output_dir_arg, recursive = TRUE, showWarnings = FALSE)
output_dir <- normalizePath(output_dir_arg, winslash = "/", mustWork = TRUE)
utils::write.csv(
  seed_coverage, file.path(output_dir, "coverage_by_seed.csv"), row.names = FALSE
)
utils::write.csv(
  coverage_summary, file.path(output_dir, "coverage_by_nominal_mass.csv"),
  row.names = FALSE
)
utils::write.csv(
  coverage_95, file.path(output_dir, "coverage_95_summary.csv"), row.names = FALSE
)
utils::write.csv(
  input_manifest, file.path(output_dir, "input_manifest.csv"), row.names = FALSE
)

commits <- unique(input_manifest$commit_hash)
r_versions <- unique(input_manifest$r_version)
metadata_lines <- c(
  paste0("generated_at_utc=", format(Sys.time(), tz = "UTC", usetz = TRUE)),
  paste0("script=", script_path()),
  paste0("script_sha256=", unname(tools::sha256sum(script_path()))),
  paste0("input_dir=", input_dir),
  paste0("output_dir=", output_dir),
  paste0("jobs=", expected_jobs),
  paste0("seeds_per_cell=", expected_seeds),
  paste0("design_cells=", expected_jobs / expected_seeds),
  "nominal_masses=0.01 through 0.99",
  "coverage_groups=all_pooled,group0,group1,equal_mixture",
  "all_pooled=unweighted mean across all observations",
  "equal_mixture=(group0 coverage + group1 coverage)/2",
  "group_index_assumption=columns 1:n0 are group0 and remaining n1 columns are group1, as constructed by the hashed canonical runners",
  "uncertainty=sample SD and Monte Carlo SE across the 50 seed-level coverage rates",
  paste0("commit_hashes=", paste(commits, collapse = ",")),
  paste0("input_r_versions=", paste(r_versions, collapse = ";")),
  paste0("summary_r_version=", R.version.string),
  paste0("locale=", paste(Sys.getlocale(), collapse = ";"))
)
writeLines(metadata_lines, file.path(output_dir, "summary_metadata.txt"), useBytes = TRUE)

cat("Coverage summary created: ", output_dir, "\n", sep = "")
