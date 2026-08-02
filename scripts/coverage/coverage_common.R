coverage_parse_args <- function(args) {
  result <- list()
  for (arg in args) {
    if (!startsWith(arg, "--") || !grepl("=", arg, fixed = TRUE)) {
      stop("Arguments must use --name=value syntax: ", arg)
    }
    pair <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1]]
    name <- pair[[1]]
    value <- paste(pair[-1], collapse = "=")
    result[[name]] <- value
  }
  result
}

coverage_arg <- function(args, name, default = NULL, required = FALSE) {
  value <- args[[name]]
  if (is.null(value)) {
    if (isTRUE(required)) {
      stop("Missing required argument --", name, "=...")
    }
    return(default)
  }
  value
}

coverage_int <- function(value, name, minimum = 1L) {
  if (length(value) != 1L || !grepl("^[0-9]+$", value)) {
    stop("--", name, " must be an integer >= ", minimum)
  }
  parsed <- suppressWarnings(as.integer(value))
  if (length(parsed) != 1L || is.na(parsed) || parsed < minimum) {
    stop("--", name, " must be an integer >= ", minimum)
  }
  parsed
}

coverage_bool <- function(value, name) {
  normalized <- tolower(value)
  if (normalized %in% c("true", "1", "yes")) {
    return(TRUE)
  }
  if (normalized %in% c("false", "0", "no")) {
    return(FALSE)
  }
  stop("--", name, " must be true or false")
}

coverage_mode <- function(args) {
  mode <- coverage_arg(args, "mode", default = "smoke")
  if (!mode %in% c("smoke", "canonical")) {
    stop("--mode must be smoke or canonical")
  }
  if (identical(mode, "canonical") &&
      !identical(coverage_arg(args, "confirm-canonical", default = ""), "YES")) {
    stop("Canonical mode requires --confirm-canonical=YES")
  }
  mode
}

coverage_script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(file_arg)) {
    stop("This script must be run with Rscript")
  }
  normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE)
}

coverage_project_root <- function(script_path) {
  normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = TRUE)
}

coverage_load_batts <- function(batts_lib) {
  normalized <- normalizePath(batts_lib, mustWork = TRUE)
  if (!dir.exists(file.path(normalized, "BATTS"))) {
    stop("BATTS is not installed under --batts-lib: ", normalized)
  }
  .libPaths(c(normalized, .libPaths()))
  suppressPackageStartupMessages(library(BATTS, lib.loc = normalized))
  normalized
}

coverage_bart_settings <- function(mode, family) {
  if (identical(mode, "smoke")) {
    return(list(
      num_trees = 20L,
      size_burnin = 20L,
      size_backfitting = 40L,
      lambda_0 = 5L
    ))
  }
  if (identical(family, "1d")) {
    return(list(
      num_trees = 200L,
      size_burnin = 1000L,
      size_backfitting = 2000L,
      lambda_0 = 5L
    ))
  }
  list(
    num_trees = 200L,
    size_burnin = 2000L,
    size_backfitting = 1000L,
    lambda_0 = 5L
  )
}

coverage_compute <- function(log_ratio_draws, truth,
                             quant_probs = seq(0.005, 0.995, by = 0.005)) {
  if (!is.matrix(log_ratio_draws) || nrow(log_ratio_draws) != length(truth)) {
    stop("Posterior draws and truth have incompatible dimensions")
  }
  credible_intervals <- apply(
    log_ratio_draws,
    1,
    stats::quantile,
    probs = quant_probs
  )
  n_probs <- length(quant_probs)
  lower_index <- seq_len(floor(n_probs / 2L))
  upper_index <- n_probs - lower_index + 1L
  symmetry_error <- max(abs(
    quant_probs[upper_index] - (1 - quant_probs[lower_index])
  ))
  if (symmetry_error > 1e-12) {
    stop("Credible-interval probability pairs are not symmetric")
  }
  truth_matrix <- matrix(
    truth,
    nrow = length(lower_index),
    ncol = length(truth),
    byrow = TRUE
  )
  included <-
    (credible_intervals[lower_index, , drop = FALSE] < truth_matrix) &
    (truth_matrix < credible_intervals[upper_index, , drop = FALSE])

  list(
    nominal_mass = 1 - 2 * quant_probs[lower_index],
    coverage_rate_BAT = rowMeans(included),
    is_included_mat = included,
    lower_prob = quant_probs[lower_index],
    upper_prob = quant_probs[upper_index],
    upper_index = upper_index,
    symmetry_max_abs = symmetry_error
  )
}

coverage_fit <- function(data, group_labels, truth, settings, reset_seed = NULL,
                         output_ensembles = FALSE) {
  if (!is.matrix(data) || !nrow(data) || !ncol(data) || any(!is.finite(data))) {
    stop("Data must be a nonempty finite numeric matrix")
  }
  if (length(group_labels) != nrow(data) ||
      !all(group_labels %in% c(0L, 1L)) ||
      !all(c(0L, 1L) %in% group_labels)) {
    stop("Group labels must contain both 0 and 1 and match the data rows")
  }
  if (length(truth) != nrow(data) || any(!is.finite(truth))) {
    stop("Finite truth values are required for every observation")
  }
  if (!is.null(reset_seed)) {
    set.seed(reset_seed)
  }
  timing_start <- proc.time()
  fit <- BATTS::batts(
    data = data,
    group_labels = group_labels,
    num_trees = settings$num_trees,
    margin_scale = 0.1,
    size_burnin = settings$size_burnin,
    size_backfitting = settings$size_backfitting,
    output_BART_ensembles = output_ensembles,
    lambda_0 = settings$lambda_0,
    quiet = TRUE,
    update_lambda = FALSE
  )
  if (!is.matrix(fit$balance_weight_BART_data) ||
      any(!is.finite(fit$balance_weight_BART_data)) ||
      any(fit$balance_weight_BART_data <= 0)) {
    stop("BATTS returned nonpositive or nonfinite posterior weights")
  }
  timing <- proc.time() - timing_start
  log_ratio_draws <- 2 * log(fit$balance_weight_BART_data)
  coverage <- coverage_compute(log_ratio_draws, truth)
  coverage$posterior_mean <- rowMeans(log_ratio_draws)
  coverage$fit_c <- fit$c
  coverage$n_observations <- nrow(data)
  coverage$n_draws <- ncol(log_ratio_draws)
  coverage$timing <- c(
    user_seconds = unname(timing[["user.self"]]),
    system_seconds = unname(timing[["sys.self"]]),
    elapsed_seconds = unname(timing[["elapsed"]])
  )
  coverage
}

coverage_git_commit <- function(project_root) {
  output <- tryCatch(
    system2(
      "git",
      c(
        "-c",
        paste0("safe.directory=", gsub("\\\\", "/", project_root)),
        "-C",
        project_root,
        "rev-parse",
        "HEAD"
      ),
      stdout = TRUE,
      stderr = FALSE
    ),
    error = function(error) character()
  )
  status <- attr(output, "status")
  if (!length(output) || (!is.null(status) && status != 0L)) {
    return(NA_character_)
  }
  output[[1]]
}

coverage_require_clean_git <- function(project_root, mode) {
  if (!identical(mode, "canonical")) {
    return(invisible(TRUE))
  }
  output <- tryCatch(
    system2(
      "git",
      c(
        "-c",
        paste0("safe.directory=", gsub("\\\\", "/", project_root)),
        "-C",
        project_root,
        "status",
        "--porcelain"
      ),
      stdout = TRUE,
      stderr = FALSE
    ),
    error = function(error) character()
  )
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) {
    stop("Could not verify the Git worktree state")
  }
  if (length(output)) {
    stop("Canonical mode requires a clean Git worktree")
  }
  if (is.na(coverage_git_commit(project_root))) {
    stop("Canonical mode requires a recorded Git commit")
  }
  invisible(TRUE)
}

coverage_package_versions <- function() {
  packages <- c("BATTS", "Rcpp", "RcppArmadillo", "mvtnorm", "pracma")
  versions <- vapply(packages, function(package) {
    if (!requireNamespace(package, quietly = TRUE)) {
      return(NA_character_)
    }
    as.character(utils::packageVersion(package))
  }, character(1))
  as.list(versions)
}

coverage_source_hashes <- function(files) {
  normalized <- normalizePath(files, mustWork = TRUE)
  hashes <- unname(tools::sha256sum(normalized))
  stats::setNames(as.list(hashes), basename(normalized))
}

coverage_batts_install_hashes <- function(batts_lib) {
  package_root <- file.path(batts_lib, "BATTS")
  candidates <- c(
    file.path(package_root, "DESCRIPTION"),
    file.path(package_root, "Meta", "package.rds"),
    file.path(package_root, "R", "BATTS.rdb"),
    file.path(package_root, "R", "BATTS.rdx"),
    list.files(
      file.path(package_root, "libs"),
      pattern = "^BATTS[.]dll$",
      recursive = TRUE,
      full.names = TRUE
    )
  )
  files <- unique(candidates[file.exists(candidates)])
  if (!length(files)) {
    stop("No hashable BATTS installation files were found")
  }
  hashes <- unname(tools::sha256sum(files))
  relative_names <- substring(
    normalizePath(files, winslash = "/", mustWork = TRUE),
    nchar(normalizePath(package_root, winslash = "/", mustWork = TRUE)) + 2L
  )
  stats::setNames(as.list(hashes), relative_names)
}

coverage_makeconf <- function() {
  candidates <- c(
    file.path(R.home("etc"), "Makeconf"),
    file.path(R.home("etc"), "x64", "Makeconf")
  )
  makeconf <- candidates[file.exists(candidates)][1]
  if (is.na(makeconf)) {
    return(list(file = NA_character_))
  }
  lines <- readLines(makeconf, warn = FALSE)
  keys <- c("CXX", "CXXFLAGS", "CXX11", "CXX11STD", "CXX11FLAGS")
  values <- lapply(keys, function(key) {
    match <- grep(paste0("^", key, "[[:space:]]*="), lines, value = TRUE)
    if (!length(match)) {
      return(NA_character_)
    }
    trimws(sub("^[^=]+=[[:space:]]*", "", match[[1]]))
  })
  names(values) <- tolower(keys)
  c(list(
    file = normalizePath(makeconf, mustWork = TRUE),
    sha256 = unname(tools::sha256sum(makeconf))
  ), values)
}

coverage_metadata <- function(mode, family, job_id, seed, config, project_root,
                              source_files, batts_lib) {
  list(
    schema_version = 1L,
    status = if (identical(mode, "smoke")) "SMOKE_NOT_FOR_PAPER" else "CANONICAL",
    family = family,
    job_id = job_id,
    seed = seed,
    config = config,
    commit_hash = coverage_git_commit(project_root),
    source_hashes = coverage_source_hashes(source_files),
    batts_library = batts_lib,
    batts_install_hashes = coverage_batts_install_hashes(batts_lib),
    r_version = R.version.string,
    platform = R.version$platform,
    os = as.list(Sys.info()),
    compiler = coverage_makeconf(),
    package_versions = coverage_package_versions(),
    rng_kind = RNGkind(),
    parallel_backend = "sequential",
    core_count = 1L,
    created_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
  )
}

coverage_output_paths <- function(output_root, mode, job_id) {
  output_root <- path.expand(output_root)
  mode_dir <- file.path(output_root, mode)
  dir.create(mode_dir, recursive = TRUE, showWarnings = FALSE)
  if (!dir.exists(mode_dir)) {
    stop("Could not create output directory: ", mode_dir)
  }
  list(
    result = file.path(mode_dir, paste0(job_id, ".rds")),
    manifest = file.path(mode_dir, paste0(job_id, ".manifest.rds"))
  )
}

coverage_resume_state <- function(paths, resume, expected_metadata) {
  result_exists <- file.exists(paths$result)
  manifest_exists <- file.exists(paths$manifest)
  if (!result_exists && !manifest_exists) {
    return(FALSE)
  }
  if (!isTRUE(resume)) {
    stop("Output already exists; refusing to overwrite: ", paths$result)
  }
  if (!result_exists || !manifest_exists) {
    stop("Incomplete prior output requires manual audit: ", paths$result)
  }
  manifest <- readRDS(paths$manifest)
  actual_hash <- unname(tools::sha256sum(paths$result))
  if (!identical(actual_hash, manifest$output_sha256)) {
    stop("Existing output hash does not match its manifest: ", paths$result)
  }
  identity_fields <- c(
    "schema_version", "status", "family", "job_id", "seed", "config",
    "commit_hash", "source_hashes", "batts_install_hashes"
  )
  mismatched <- identity_fields[!vapply(identity_fields, function(field) {
    identical(manifest[[field]], expected_metadata[[field]])
  }, logical(1))]
  if (length(mismatched)) {
    stop(
      "Existing output does not match the requested run identity: ",
      paste(mismatched, collapse = ", ")
    )
  }
  TRUE
}

coverage_atomic_save <- function(result, metadata, paths) {
  result_dir <- dirname(paths$result)
  result_temp <- tempfile(
    pattern = paste0(basename(paths$result), "."),
    tmpdir = result_dir,
    fileext = ".tmp"
  )
  manifest_temp <- tempfile(
    pattern = paste0(basename(paths$manifest), "."),
    tmpdir = result_dir,
    fileext = ".tmp"
  )
  on.exit(unlink(c(result_temp, manifest_temp), force = TRUE), add = TRUE)

  saveRDS(list(metadata = metadata, result = result), result_temp, compress = "xz")
  output_hash <- unname(tools::sha256sum(result_temp))
  if (!file.rename(result_temp, paths$result)) {
    stop("Atomic result rename failed: ", paths$result)
  }

  manifest <- metadata
  manifest$output_file <- basename(paths$result)
  manifest$output_sha256 <- output_hash
  manifest$output_bytes <- file.info(paths$result)$size
  saveRDS(manifest, manifest_temp, compress = "xz")
  if (!file.rename(manifest_temp, paths$manifest)) {
    stop("Atomic manifest rename failed: ", paths$manifest)
  }
  invisible(manifest)
}
